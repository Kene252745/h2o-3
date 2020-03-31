package water.rapids;

// Since we have a single key field in H2O (different to data.table), bmerge() becomes a lot simpler (no
// need for recursion through join columns) with a downside of transfer-cost should we not need all the key.

import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.parser.BufferedString;
import water.util.ArrayUtils;
import water.util.Log;

import java.math.BigInteger;
import java.util.Arrays;
import static java.math.BigInteger.ONE;

import static water.rapids.SingleThreadRadixOrder.getSortedOXHeaderKey;

class BinaryMerge extends DTask<BinaryMerge> {
  long _numRowsInResult=0;  // returned to caller, so not transient
  int _chunkSizes[]; // TODO:  only _chunkSizes.length is needed by caller, so return that length only
  double _timings[];

  private transient long _ret1st[/*n2GB*/][];  // The row number of the first right table's index key that matches
  private transient long _retLen[/*n2GB*/][];   // How many rows does it match to?

  final FFSB _leftSB, _riteSB;
  final boolean _onlyLeftFrame;  // denote if only left frame is available which implies sorting.
  private transient KeyOrder _leftKO, _riteKO;

  private final int _numJoinCols;
  private transient long _leftFrom;
  private transient int _retBatchSize; // no need to match batchsize of RadixOrder.

  private final boolean _allLeft, _allRight;
  private boolean[] _stringCols;
  private boolean[] _intCols;

  // does any left row match to more than 1 right row?  If not, can allocate
  // and loop more efficiently, and mark the resulting key'd frame with a
  // 'unique' index.  //   TODO: implement
  private transient boolean _oneToManyMatch = false;

  // Data which is duplicated left and rite, but only one copy is needed
  // per-map.  This data is made in the constructor and shallow-copy shared
  // around the cluster.
  static class FFSB extends Iced<FFSB> {
    private final Frame _frame;
    private final Vec _vec;
    // fast lookups to save repeated calls to node.index() which calls
    // binarysearch within it.
    private final int _chunkNode[]; // Chunk homenode index
    final int _msb;
    private final int _shift;
    private final BigInteger _base[]; // the col.min() of each column in the key
    private final int _fieldSizes[]; // the widths of each column in the key
    private final int _keySize; // the total width in bytes of the key, sum of field sizes

    FFSB( Frame frame, int msb, int shift, int fieldSizes[], BigInteger base[]) {
      assert -1<=msb && msb<=255; // left ranges from 0 to 255, right from -1 to 255
      _frame = frame;
      _msb = msb;
      _shift = shift;
      _fieldSizes = fieldSizes;
      _keySize = ArrayUtils.sum(fieldSizes);
      _base = base;
      // Create fast lookups to go from chunk index to node index of that chunk
      Vec vec = _vec = frame.anyVec();
      _chunkNode = vec==null ? null : MemoryManager.malloc4(vec.nChunks());
      if( vec == null ) return; // Zero-columns for Sort
      for( int i=0; i<_chunkNode.length; i++ )
        _chunkNode[i] = vec.chunkKey(i).home_node().index();
    }
    
    long min() {
      return BigInteger.valueOf(((long)_msb) << _shift).add(_base[0].subtract(ONE)).longValue();
    }
    long max() {
      return BigInteger.valueOf(((long)_msb+1) << _shift).add(_base[0].subtract(ONE).subtract(ONE)).longValue();
    }
  }

  // In X[Y], 'left'=i and 'right'=x
  BinaryMerge(FFSB leftSB, FFSB riteSB, boolean allLeft) {
    assert riteSB._msb!=-1 || allLeft;
    _leftSB = leftSB;
    _riteSB = riteSB;
    _onlyLeftFrame = (_leftSB._frame.numCols() > 0 && _riteSB._frame.numCols()==0);
    // the number of columns in the key i.e. length of _leftFieldSizes and _riteSB._fieldSizes
    _numJoinCols = Math.min(_leftSB._fieldSizes.length, _riteSB._fieldSizes.length);
    _allLeft = allLeft;
    _allRight = false;  // TODO: pass through
    int columnsInResult = (_leftSB._frame == null?0:_leftSB._frame.numCols()) +
      (_riteSB._frame == null?0:_riteSB._frame.numCols())-_numJoinCols;
    _stringCols = MemoryManager.mallocZ(columnsInResult);
    _intCols = MemoryManager.mallocZ(columnsInResult);
    // check left frame first
    if (_leftSB._frame!=null) {
      for (int col=0; col <_numJoinCols; col++) {
        if (_leftSB._frame.vec(col).isInt())
          _intCols[col] = true;
      }
      for (int col = _numJoinCols; col < _leftSB._frame.numCols(); col++) {
        if (_leftSB._frame.vec(col).isString())
          _stringCols[col] = true;

        if( _leftSB._frame.vec(col).isInt() )
          _intCols[col] = true;
      }
    }
    // check right frame next
    if (_riteSB._frame != null) {
      int colOffset = _leftSB._frame==null?0:_leftSB._frame.numCols()-_numJoinCols;
      for (int col = _numJoinCols; col < _riteSB._frame.numCols(); col++) {
        if (_riteSB._frame.vec(col).isString())
          _stringCols[col + colOffset] = true;
        if( _riteSB._frame.vec(col).isInt() )
          _intCols[col+colOffset] = true;
      }
    }
  }


  @Override
  public void compute2() {
    _timings = MemoryManager.malloc8d(20);
    long t0 = System.nanoTime();

    SingleThreadRadixOrder.OXHeader leftSortedOXHeader = DKV.getGet(getSortedOXHeaderKey(/*left=*/true, _leftSB._msb));
    if (leftSortedOXHeader == null) {
      if( !_allRight ) { tryComplete(); return; }
      throw H2O.unimpl();  // TODO pass through _allRight and implement
    }
    _leftKO = new KeyOrder(leftSortedOXHeader);

    SingleThreadRadixOrder.OXHeader rightSortedOXHeader = DKV.getGet(getSortedOXHeaderKey(/*left=*/false, _riteSB._msb));
    //if (_riteSB._msb==-1) assert _allLeft && rightSortedOXHeader == null; // i.e. it's known nothing on right can join
    if (rightSortedOXHeader == null) {
      if( !_allLeft ) { tryComplete(); return; }
      // enables general case code to run below without needing new special case code
      rightSortedOXHeader = new SingleThreadRadixOrder.OXHeader(0, 0, 0);
    }
    _riteKO = new KeyOrder(rightSortedOXHeader);

    // get left batches
    _leftKO.initKeyOrder(_leftSB._msb,/*left=*/true);
    final long leftN = leftSortedOXHeader._numRows; // number of leftframe rows to fetch for leftMSB
    assert leftN >= 1;

    // get right batches
    _riteKO.initKeyOrder(_riteSB._msb, /*left=*/false);
    final long rightN = rightSortedOXHeader._numRows;

    _timings[0] += (System.nanoTime() - t0) / 1e9;


    // Now calculate which subset of leftMSB and which subset of rightMSB we're
    // joining here by going into the detail of the key values present rather
    // than the extents of the range (the extents themselves may not be
    // present).

    // We see where the right extents occur in the left keys present; and if
    // there is an overlap we find the full extent of the overlap on the left
    // side (nothing less).

    // We only _need_ do this for left outer join otherwise we'd end up with
    // too many no-match left rows.

    // We'll waste allocating the retFirst and retLen vectors though if only a
    // small overlap is needed, so for that reason it's useful to restrict size
    // of retFirst and retLen even for inner join too.

    // Find left and right MSB extents in terms of the key boundaries they represent
    // _riteSB._msb==-1 indicates that no right MSB should be looked at
    final long leftMin = _leftSB.min();  // the minimum possible key value in this bucket
    final long leftMax = _leftSB.max();  // the maximum possible key value in this bucket
    // if _riteSB._msb==-1 then the values in riteMin and riteMax here are redundant and not used
    final long riteMin = _riteSB._msb==-1 ? -1 : _riteSB.min();  // the minimum possible key value in this bucket
    final long riteMax = _riteSB._msb==-1 ? -1 : _riteSB.max();  // the maximum possible key value in this bucket

    // _leftFrom and leftTo refers to the row indices to perform merging/search for each MSB value
    _leftFrom =   (_riteSB._msb==-1 || leftMin>=riteMin || (_allLeft && _riteSB._msb==0  )) ? -1    : bsearchLeft(riteMin, /*retLow*/true , leftN);
    long leftTo = (_riteSB._msb==-1 || leftMax<=riteMax || (_allLeft && _riteSB._msb==255)) ? leftN : bsearchLeft(riteMax, /*retLow*/false, leftN);
    // The (_allLeft && rightMSB==0) part is to include those keys in that
    // leftMSB just below the right base.  They won't be caught by rightMSBs to
    // the left because there are no more rightMSBs below 0.  Only when
    // _allLeft do we need to create NA match for them.  They must be created
    // in the same MSB/MSB pair along with the keys that may match the very
    // lowest right keys, because stitching assumes unique MSB/MSB pairs.

    long retSize = leftTo - _leftFrom - 1;   // since leftTo and leftFrom are 1 outside the extremes
    assert retSize >= 0; // retSize is number of rows to include in final merged frame
    if (retSize==0) { tryComplete(); return; } // nothing can match, even when allLeft

    _retBatchSize = 1048576;   // must set to be the same from RadixOrder.java
    int retNBatch = (int)((retSize - 1) / _retBatchSize + 1);
    int retLastSize = (int)(retSize - (retNBatch - 1) * _retBatchSize);

    _ret1st = new long[retNBatch][];
    _retLen = new long[retNBatch][];
    for( int b=0; b<retNBatch; b++) {
      _ret1st[b] = MemoryManager.malloc8(b==retNBatch-1 ? retLastSize : _retBatchSize);
      _retLen[b] = MemoryManager.malloc8(b==retNBatch-1 ? retLastSize : _retBatchSize);
    }

    // always look at the whole right bucket.  Even though in types -1 and 1,
    // we know range is outside so nothing should match.  if types -1 and 1 do
    // occur, they only happen for leftMSB 0 and 255, and will quickly resolve
    // to no match in the right bucket via bmerge
    t0 = System.nanoTime();
    binaryMergeOneMSB(_leftFrom, leftTo, -1, rightN);
    _timings[1] += (System.nanoTime() - t0) / 1e9;

    if (_allLeft) {
      assert _leftKO.numRowsToFetch() == retSize;
    } else {
      long tt = 0;
      for( long[] retFirstx : _ret1st )    // i.e. sum(_ret1st>0) in R
        for( long rF : retFirstx )
          tt += (rF > 0) ? 1 : 0;
      // TODO: change to tt.privateAssertMethod() containing the loop above to
      //       avoid that loop when asserts are off, or accumulate the tt
      //       inside the merge_r, somehow
      assert tt <= retSize;
      assert _leftKO.numRowsToFetch() == tt;
    }

    if (_numRowsInResult > 0) createChunksInDKV();

    // TODO: set 2 Frame and 2 int[] to NULL at the end of compute2 to save
    // some traffic back, but should be small and insignificant
    // TODO: recheck transients or null out here before returning
    tryComplete();
  }

  // Holder for Key & Order info
  private static class KeyOrder {
    public final transient long _batchSize;
    private final transient byte _key  [/*n2GB*/][/*i mod 2GB * _keySize*/];
    private final transient long _order[/*n2GB*/][/*i mod 2GB * _keySize*/];
    private final transient long _perNodeNumRowsToFetch[];

    KeyOrder( SingleThreadRadixOrder.OXHeader sortedOXHeader ) {
      _batchSize = sortedOXHeader._batchSize;
      final int nBatch = sortedOXHeader._nBatch;
      _key   = new byte[nBatch][];
      _order = new long[nBatch][];
      _perNodeNumRowsToFetch = new long[H2O.CLOUD.size()];
    }

    void initKeyOrder( int msb, boolean isLeft ) {
      for( int b=0; b<_key.length; b++ ) {
        Value v = DKV.get(SplitByMSBLocal.getSortedOXbatchKey(isLeft, msb, b));
        SplitByMSBLocal.OXbatch ox = v.get(); //mem version (obtained from remote) of the Values gets turned into POJO version
        v.freeMem(); //only keep the POJO version of the Value
        _key  [b] = ox._x;
        _order[b] = ox._o;
      }
    }
    long numRowsToFetch() { return ArrayUtils.sum(_perNodeNumRowsToFetch); }
    // Do a mod/div long _order array lookup
    long at8order( long idx ) { return _order[(int)(idx / _batchSize)][(int)(idx % _batchSize)]; }

    long[][] fillPerNodeRows( int i, final int batchSizeLong) {
     // final int batchSizeLong = 256*1024*1024 / 16;  // 256GB DKV limit / sizeof(UUID)
      if( _perNodeNumRowsToFetch[i] <= 0 ) return null;
      int nbatch  = (int) ((_perNodeNumRowsToFetch[i] - 1) / batchSizeLong + 1);  // TODO: wrap in class to avoid this boiler plate
      assert nbatch >= 1;
      int lastSize = (int) (_perNodeNumRowsToFetch[i] - (nbatch - 1) * batchSizeLong);
      assert lastSize > 0;
      long[][] res = new long[nbatch][];
      for( int b = 0; b < nbatch; b++ )
        res[b] = MemoryManager.malloc8(b==nbatch-1 ? lastSize : batchSizeLong);
      return res;
    }
  }
  
  private int keycmp(byte xss[][], long xi, KeyOrder xKO, FFSB xSB, byte yss[][], long yi, KeyOrder yKO, FFSB ySB) {
    byte xbatch[] = xss[(int)(xi / xKO._batchSize)];
    byte ybatch[] = yss[(int)(yi / yKO._batchSize)];
    int xoff = (int)(xi % xKO._batchSize) * xSB._keySize;
    int yoff = (int)(yi % yKO._batchSize) * ySB._keySize;
    long xval=0, yval=0;

    // We avoid the NewChunk compression because we want finer grain
    // compression than 1,2,4 or 8 bytes types.  In particular, a range just
    // greater than 4bn can use 5 bytes rather than 8 bytes; a 38% RAM saving
    // over the wire in that possibly common case.  Note this is tight and
    // almost branch free.
    int i=0;
    while( i<_numJoinCols && xval==yval ) { // TODO: pass i in to start at a later key column, when known
      int xlen = xSB._fieldSizes[i];
      int ylen = ySB._fieldSizes[i];
      xval = xbatch[xoff] & 0xFFL; while (xlen>1) { xval <<= 8; xval |= xbatch[++xoff] & 0xFFL; xlen--; } xoff++;
      yval = ybatch[yoff] & 0xFFL; while (ylen>1) { yval <<= 8; yval |= ybatch[++yoff] & 0xFFL; ylen--; } yoff++;

      xval = xval==0 ? Long.MIN_VALUE : updateVal(xval,xSB._base[i]);
      yval = yval==0 ? Long.MIN_VALUE : updateVal(yval,ySB._base[i]);

      i++;
    }

    // The magnitude of the difference is used for limiting staleness in a
    // rolling join, capped at Integer.MAX|(MIN+1).  Roll's type is chosen to
    // be int so staleness can't be requested over int's limit.
    // Same return value as strcmp in C. <0 => xi<yi.
    long diff = xval-yval;  // could overflow even in long; e.g. joining to a prevailing NA, or very large gaps O(2^62)

    if (BigInteger.valueOf(xval).subtract(BigInteger.valueOf(yval)).bitLength() > 64)
      Log.warn("Overflow in BinaryMerge.java");  // detects overflow

    if (xval>yval) {        // careful not diff>0 here due to overflow
      return( (diff<0 | diff>Integer.MAX_VALUE  ) ? Integer.MAX_VALUE   : (int)diff);
    } else {
      return( (diff>0 | diff<Integer.MIN_VALUE+1) ? Integer.MIN_VALUE+1 : (int)diff);
    }
  }

  private long updateVal(Long oldVal, BigInteger baseD) {
    // we know oldVal is not zero
    BigInteger xInc = baseD.add(BigInteger.valueOf(oldVal).subtract(ONE));
    if (xInc.bitLength() > 64) {
      Log.warn("Overflow in BinaryMerge.java");
      return oldVal;  // should have died sooner or later
    } else
      return xInc.longValue();
  }

  // binary search to the left MSB in the 1st column only
  private long bsearchLeft(long x, boolean returnLow, long upp)  {
    long low = -1;
    while (low < upp - 1) {
      long mid = low + (upp - low) / 2;
      byte keyBatch[] = _leftKO._key[(int)(mid / _leftKO._batchSize)];
      int off = (int)(mid % _leftKO._batchSize) * _leftSB._keySize;
      int len = _leftSB._fieldSizes[0];
      long val = keyBatch[off] & 0xFFL;
      while( len>1 ) {
        val <<= 8; val |= keyBatch[++off] & 0xFFL; len--;
      }

      val = val==0 ? Long.MIN_VALUE : updateVal(val,_leftSB._base[0]);
      if (x<val || (x==val && returnLow)) {
        upp = mid;
      } else {
        low = mid;
      }
    }
    return returnLow ? low : upp;
  }

  /***
   * For a specific MSB, if we can find MSBs in the left frame and rite frame that contains values in the MSB range,
   * we will try to find match between the left frame and rite frame and include those rows in the final merged frame.
   * If no match is found between the two frames but allLeft = true, we will go ahead and just include all the rows
   * having the same MSB in the left frame into the final merged frame.
   * 
   * If allLeft is true, we will always iterate over the leftFrame keys and use binary search over the rite frame keys.
   * We will count all duplicate keys in the left and rite frames as well.  During the search process, we should be 
   * able to shrink the range of rite frame keys to search over as we gain more knowledge over the relative key 
   * sizes between the two frames and all keys are sorted already.
   * 
   * However, when allLeft is false, we will iterate over the frames that has the smallest number of keys instead. 
   * Hence, in this case, we can search over either the left or rite frame depending on the number of rows to be 
   * searched through.  
   * 
   * @param leftLowIn: left frame lowest row number minus 1
   * @param leftUppIn: number of rows in left frame with specific MSB
   * @param riteLowIn: rite frame lowest row number minus 1 with specific MSB
   * @param riteUppIn: number of rows in rite frame with specific MSB
   *                  
   */
  private void binaryMergeOneMSB(long leftLowIn, long leftUppIn, long riteLowIn, long riteUppIn) {
    if (!_allLeft && riteUppIn == 0) return;  // no merging possible with empty rite frame here
    boolean leftFrameIterate = _allLeft ? true : ((leftUppIn - leftLowIn) > (riteUppIn - riteLowIn) ? false : true);
    long iterIndex = leftFrameIterate ? leftLowIn + 1 : riteLowIn + 1;
    long iterUppIn = leftFrameIterate ? leftUppIn : riteUppIn;
    long binarySearchUppIn = leftFrameIterate ? riteUppIn : leftUppIn;
    long iterHigh, binarySearchLow, binarySearchHigh;  // temp variables to perform search
    MidSearchInfo binarySearchInfo = new MidSearchInfo(0, leftFrameIterate ? riteLowIn : leftLowIn,
            leftFrameIterate ? riteUppIn : leftUppIn); // store match row, riteLow, riteUpp
    int binarySearchLen = 0;  // number of iterate Frame/binarySearch frame rows to be included to merged frame

    while (iterIndex < iterUppIn) { // for each left row, find matches in the right frame using binary search
      bsearchRiteMatch(iterIndex, binarySearchInfo, leftFrameIterate); // try to find match for leftFrame at leftIndex in riteFrame
      // extend index to include/skip over duplicates in iterate frame
      iterHigh = iterIndex + 1;
      while (iterHigh < iterUppIn && frameKeyEqual(leftFrameIterate ? _leftKO._key : _riteKO._key, iterIndex, iterHigh,
              leftFrameIterate ? _leftKO : _riteKO, leftFrameIterate ? _leftSB : _riteSB)) {
        iterHigh++; // find iterate frame duplicates
      }

      if (binarySearchInfo._matchIndex >= 0) {  // match found in riteFrame, 
        // find duplicates in binarysearch frame, could be lower or higher than midSearchInd[0]
        binarySearchLow = binarySearchInfo._matchIndex;
        long tempValue = binarySearchInfo._matchIndex - 1;
        while (tempValue >= 0 && frameKeyEqual(leftFrameIterate ? _riteKO._key : _leftKO._key, binarySearchInfo._matchIndex,
                tempValue, leftFrameIterate ? _riteKO : _leftKO, leftFrameIterate ? _riteSB : _leftSB)) {
          binarySearchLow--;
          tempValue--;
        }

        binarySearchHigh = binarySearchInfo._matchIndex + 1;
        while (binarySearchHigh < binarySearchUppIn && frameKeyEqual(leftFrameIterate ? _riteKO._key : _leftKO._key,
                binarySearchInfo._matchIndex, binarySearchHigh, leftFrameIterate ? _riteKO : _leftKO,
                leftFrameIterate ? _riteSB : _leftSB)) {
          binarySearchHigh++;
        }
        binarySearchLen = (int) (binarySearchHigh - binarySearchLow);
        binarySearchInfo._lowSearchIndex = binarySearchHigh; // shrink binarysearch frame range
        binarySearchInfo._matchIndex = binarySearchLow;
      } else {  // no match, adjust binarySearch upper range, set all Len to 0
        binarySearchLen = 0;
      }
      binarySearchInfo._upperSearchIndex = leftFrameIterate ? riteUppIn : leftUppIn;
      // organize merged frame for each iteration over iterate frame
      populate_ret1st_retLen((int) (iterHigh - iterIndex), binarySearchLen, iterIndex, binarySearchInfo._matchIndex, leftFrameIterate);
      iterIndex = iterHigh; // next row index to iterate over
    }
  }
  
  private class MidSearchInfo {
    long _matchIndex;
    long _lowSearchIndex;
    long _upperSearchIndex;
    
    public MidSearchInfo(long matchI, long lowI, long highI) {
      _matchIndex = matchI;
      _lowSearchIndex = lowI;
      _upperSearchIndex = highI;
    }
  }
  
  private void populate_ret1st_retLen(int iterLen, int binarySearchLen, long iterIndex, long matchIndex, 
                                      boolean leftFrameIterate) {
    if (!_allLeft && (iterLen==0 || binarySearchLen==0)) return; 
    if (_allLeft && iterLen > 1) _oneToManyMatch=true;  // duplicate keys found in leftFrame
    _numRowsInResult += Math.max(1,iterLen)*Math.max(1,binarySearchLen);  // add contribution to final merged frame
    long leftLow = leftFrameIterate?iterIndex:matchIndex;
    long leftHigh = leftFrameIterate?(iterIndex+iterLen):(matchIndex+binarySearchLen);
    long rLow = leftFrameIterate?matchIndex:iterIndex;
    // iterate over left frame and add rows to final merged frame
    for (long leftInd = leftLow; leftInd < leftHigh; leftInd++) { // allocate _ret1st and _retlen according to # of left frame rows
      long globalRowNumber = _leftKO.at8order(leftInd);
      int chkIdx = _leftSB._vec.elem2ChunkIdx(globalRowNumber);// obtain chunk index of row of interest
      _leftKO._perNodeNumRowsToFetch[_leftSB._chunkNode[chkIdx]]++;
      if (_allLeft && binarySearchLen==0)
        continue; // no need to fill out _ret1st and _retLen if there is no match in rite frame
      long fLocationIndex = leftInd-(_leftFrom+1);
      final int batchIndex = (int) fLocationIndex/_retBatchSize;
      final int batchOffset = (int) fLocationIndex%_retBatchSize;
      _ret1st[batchIndex][batchOffset] = rLow+1;  // 0 is to denote no match between left and rite frame
      _retLen[batchIndex][batchOffset] = leftFrameIterate?binarySearchLen:iterLen;
    }
    // fill in rite frame and add rows to final merged frame if appropriate
    if (!(_allLeft && binarySearchLen==0)) {
      long riteLen = leftFrameIterate?binarySearchLen:iterLen; // number of duplication in right frame
      for (long riteInd = 0; riteInd < riteLen; riteInd++) {
        long loc = rLow + riteInd;
        long globalRowNumber = _riteKO.at8order(loc);
        int chkIdx = _riteSB._vec.elem2ChunkIdx(globalRowNumber);
        _riteKO._perNodeNumRowsToFetch[_riteSB._chunkNode[chkIdx]]++;
      }
    }
  }

  private boolean frameKeyEqual(byte x[][], long xi, long yi, KeyOrder fKO, FFSB fSB) {
    byte xbatch[] = x[(int)(xi / fKO._batchSize)];
    byte ybatch[] = x[(int)(yi / fKO._batchSize)];
    int xoff = (int)(xi % fKO._batchSize) * fSB._keySize;
    int yoff = (int)(yi % fKO._batchSize) * fSB._keySize;
    int i=0;
    while (i<fSB._keySize && xbatch[xoff++] == ybatch[yoff++]) i++;
    return(i==fSB._keySize);
  }

  private void bsearchRiteMatch(long iterIndex, MidSearchInfo midSearchInd, boolean leftFIter) {
    while (midSearchInd._lowSearchIndex < midSearchInd._upperSearchIndex) {
      midSearchInd._matchIndex=midSearchInd._lowSearchIndex+(midSearchInd._upperSearchIndex -midSearchInd._lowSearchIndex)/2;
      if (midSearchInd._matchIndex<0) // skip over case of -1,0,1, binary search frame contains 0 entry to compare
        return;
      int cmp = leftFIter?keycmp(_leftKO._key, iterIndex, _leftKO, _leftSB, _riteKO._key, midSearchInd._matchIndex, _riteKO, _riteSB):
              keycmp(_riteKO._key, iterIndex, _riteKO, _riteSB, _leftKO._key, midSearchInd._matchIndex, _leftKO, _leftSB);// 0 equal, <0 smaller, >0 bigger
      if (cmp < 0) { // iterate frame value lower than binarysearch value, shrink binarysearch frame index down to midRite
        midSearchInd._upperSearchIndex =midSearchInd._matchIndex;
      } else if (cmp > 0) { // iterate frame value higher than binarysearch value, move up binarysearch frame index
        midSearchInd._lowSearchIndex=midSearchInd._matchIndex+1;
      } else {  // iterate and binarysearch frame key match
        return;
      }
    }
    midSearchInd._matchIndex=-1;
  }

  private void createChunksInDKV() {
    // Collect all matches
    // Create the final frame (part) for this MSB combination
    // Cannot use a List<Long> as that's restricted to 2Bn items and also isn't an Iced datatype
    long t0 = System.nanoTime(), t1;

    final int cloudSize = H2O.CLOUD.size();
    final long perNodeRightRows[][][] = new long[cloudSize][][];
    final long perNodeLeftRows [][][] = new long[cloudSize][][];

    // Allocate memory to split this MSB combn's left and right matching rows
    // into contiguous batches sent to the nodes they reside on
    for( int i = 0; i < cloudSize; i++ ) {
      perNodeRightRows[i] = _riteKO.fillPerNodeRows(i, (int) _riteKO._batchSize);
      perNodeLeftRows [i] = _leftKO.fillPerNodeRows(i, (int) _leftKO._batchSize);
    }
    _timings[2] += ((t1=System.nanoTime()) - t0) / 1e9; t0=t1;

    // Loop over _ret1st and _retLen and populate the batched requests for
    // each node helper.  _ret1st and _retLen are the same shape
    final long perNodeRightLoc[] = MemoryManager.malloc8(cloudSize);
    final long perNodeLeftLoc [] = MemoryManager.malloc8(cloudSize);
    chunksPopulatePerNode(perNodeLeftLoc,perNodeLeftRows,perNodeRightLoc,perNodeRightRows);
    _timings[3] += ((t1=System.nanoTime()) - t0) / 1e9; t0=t1;

    // Create the chunks for the final frame from this MSB pair.

    // 16 bytes for each UUID (biggest type). Enum will be long (8). TODO: How is non-Enum 'string' handled by H2O?
    final int batchSizeUUID = _retBatchSize;
    final int nbatch = (int) ((_numRowsInResult-1)/batchSizeUUID +1);  // TODO: wrap in class to avoid this boiler plate
    assert nbatch >= 1;
    final int lastSize = (int) (_numRowsInResult - (nbatch-1)*batchSizeUUID);
    assert lastSize > 0;
    final int numLeftCols = _leftSB._frame.numCols();
    final int numColsInResult = _leftSB._frame.numCols() + _riteSB._frame.numCols() - _numJoinCols;
    final double[][][] frameLikeChunks = new double[numColsInResult][nbatch][]; //TODO: compression via int types
    final long[][][] frameLikeChunksLongs = new long[numColsInResult][nbatch][]; //TODO: compression via int types
    BufferedString[][][] frameLikeChunks4Strings = new BufferedString[numColsInResult][nbatch][]; // cannot allocate before hand
    _chunkSizes = new int[nbatch];

    final GetRawRemoteRows grrrsLeft[][] = new GetRawRemoteRows[cloudSize][];
    final GetRawRemoteRows grrrsRite[][] = new GetRawRemoteRows[cloudSize][];

    if (_onlyLeftFrame) { // sorting only
      long[] resultLeftLocPrevlPrevf = new long[4]; // element 0 store resultLoc, element 1 store leftLoc
      resultLeftLocPrevlPrevf[1] = _leftFrom; // sweep through left table along the sorted row locations.
      resultLeftLocPrevlPrevf[0] = 0;
      resultLeftLocPrevlPrevf[2] = -1;
      resultLeftLocPrevlPrevf[3] = -1;

      for (int b = 0; b < nbatch; b++) {
        allocateFrameLikeChunks(b, nbatch, lastSize, batchSizeUUID, frameLikeChunks, frameLikeChunks4Strings, 
                frameLikeChunksLongs, numColsInResult);

        // Now loop through _ret1st and _retLen and populate
        chunksPopulateRetFirst(perNodeLeftRows, resultLeftLocPrevlPrevf, b, numColsInResult,
                numLeftCols, perNodeLeftLoc, grrrsLeft, frameLikeChunks,
                frameLikeChunks4Strings, frameLikeChunksLongs);
        _timings[10] += ((t1 = System.nanoTime()) - t0) / 1e9;
        t0 = t1;
        // compress all chunks and store them
        chunksCompressAndStore(b, numColsInResult, frameLikeChunks, frameLikeChunks4Strings, frameLikeChunksLongs);
        if (nbatch > 1) {
          cleanUpMemory(grrrsLeft, b);  // clean up memory used by grrrsLeft and grrrsRite
        }
      }
    } else { // merging
      for (int b = 0; b < nbatch; b++) { // allocate all frameLikeChunks in one shot
        allocateFrameLikeChunks(b, nbatch, lastSize, batchSizeUUID, frameLikeChunks, frameLikeChunks4Strings, 
                frameLikeChunksLongs, numColsInResult); // allocate Frame is ok
        _timings[6] += ((t1 = System.nanoTime()) - t0) / 1e9;
        t0 = t1;  // all this time is expected to be in [5]
      }

      _timings[4] += ((t1 = System.nanoTime()) - t0) / 1e9;
      t0 = t1;
      chunksGetRawRemoteRows(perNodeLeftRows, perNodeRightRows, grrrsLeft, grrrsRite); // need this one

      chunksPopulateRetFirst(batchSizeUUID, numColsInResult, numLeftCols, perNodeLeftLoc, grrrsLeft,
              perNodeRightLoc, grrrsRite, frameLikeChunks, frameLikeChunks4Strings, frameLikeChunksLongs);
      _timings[10] += ((t1 = System.nanoTime()) - t0) / 1e9;
      t0 = t1;

      chunksCompressAndStoreO(nbatch, numColsInResult, frameLikeChunks, frameLikeChunks4Strings, frameLikeChunksLongs);
    }
    _timings[11] += (System.nanoTime() - t0) / 1e9;
  }

  // compress all chunks and store them
  private void chunksCompressAndStoreO(final int nbatch, final int numColsInResult, 
                                       final double[][][] frameLikeChunks, BufferedString[][][] frameLikeChunks4String,
                                       long[][][] frameLikeChunksLong) {
    // compress all chunks and store them
    Futures fs = new Futures();
    for (int col = 0; col < numColsInResult; col++) {
      if (this._stringCols[col]) {
        for (int b = 0; b < nbatch; b++) {
          NewChunk nc = new NewChunk(null, 0);
          for (int index = 0; index < frameLikeChunks4String[col][b].length; index++)
            nc.addStr(frameLikeChunks4String[col][b][index]);
          Chunk ck = nc.compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunks4String[col][b] = null; //free mem as early as possible (it's now in the store)
        }
      } else if( _intCols[col] ) {
        for (int b = 0; b < nbatch; b++) {
          NewChunk nc = new NewChunk(null,-1);
          for(long l: frameLikeChunksLong[col][b]) {
            if( l==Long.MIN_VALUE ) 
              nc.addNA();
            else                    nc.addNum(l, 0);
          }
          Chunk ck = nc.compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunksLong[col][b] = null; //free mem as early as possible (it's now in the store)
        }
      } else {
        for (int b = 0; b < nbatch; b++) {
          Chunk ck = new NewChunk(frameLikeChunks[col][b]).compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunks[col][b] = null; //free mem as early as possible (it's now in the store)
        }
      }
    }
    fs.blockForPending();
  }
  
  private void allocateFrameLikeChunks(final int b, final int nbatch, final int lastSize, final int batchSizeUUID, 
                                       final double[][][] frameLikeChunks, 
                                       final BufferedString[][][] frameLikeChunks4Strings, 
                                       final long[][][] frameLikeChunksLongs, final int numColsInResult) {
    for (int col = 0; col < numColsInResult; col++) {  // allocate memory for frameLikeChunks for this batch
      if (this._stringCols[col]) {
        frameLikeChunks4Strings[col][b] = new BufferedString[_chunkSizes[b] = (b == nbatch - 1 ? lastSize : batchSizeUUID)];
      } else if (this._intCols[col]) {
        frameLikeChunksLongs[col][b] = MemoryManager.malloc8(_chunkSizes[b] = (b == nbatch - 1 ? lastSize : batchSizeUUID));
        Arrays.fill(frameLikeChunksLongs[col][b], Long.MIN_VALUE);
      } else {
        frameLikeChunks[col][b] = MemoryManager.malloc8d(_chunkSizes[b] = (b == nbatch - 1 ? lastSize : batchSizeUUID));
        Arrays.fill(frameLikeChunks[col][b], Double.NaN);
        // NA by default to save filling with NA for nomatches when allLeft
      }
    }
  }

  // Loop over _ret1st and _retLen and populate the batched requests for
  // each node helper.  _ret1st and _retLen are the same shape
  private void chunksPopulatePerNode( final long perNodeLeftLoc[], final long perNodeLeftRows[][][], 
                                      final long perNodeRightLoc[], final long perNodeRightRows[][][] ) {
    final int batchSizeLong = _retBatchSize;  // 256GB DKV limit / sizeof(UUID)
    long prevf = -1, prevl = -1;
    // TODO: hop back to original order here for [] syntax.
    long leftLoc=_leftFrom;  // sweep through left table along the sorted row locations.
    for (int jb=0; jb<_ret1st.length; ++jb) {              // jb = j batch
      for (int jo=0; jo<_ret1st[jb].length; ++jo) {        // jo = j offset
        leftLoc++;  // to save jb*_ret1st[0].length + jo;
        long f = _ret1st[jb][jo];  // TODO: take _ret1st[jb] outside inner loop
        long l = _retLen[jb][jo];
        if (f==0) {
          // left row matches to no right row
          assert l == 0;  // doesn't have to be 0 (could be 1 already if allLeft==true) but currently it should be, so check it
          if (!_allLeft) continue;
          // now insert the left row once and NA for the right columns i.e. left outer join
        }

        { // new scope so 'row' can be declared in the for() loop below and registerized (otherwise 'already defined in this scope' in that scope)
          // Fetch the left rows and mark the contiguous from-ranges each left row should be recycled over
          // TODO: when single node, not needed
          // TODO could loop through batches rather than / and % wastefully
          long row = _leftKO.at8order(leftLoc); // global row number of matched row in left frame
          int chkIdx = _leftSB._vec.elem2ChunkIdx(row); //binary search in espc
          int ni = _leftSB._chunkNode[chkIdx]; // node index
          long pnl = perNodeLeftLoc[ni]++;    // pnl = per node location
          perNodeLeftRows[ni][(int)(pnl/batchSizeLong)][(int)(pnl%batchSizeLong)] = row;  // ask that node for global row number row
        }
        if (f==0) continue;
        assert l > 0;
        if (prevf == f && prevl == l)
          continue;  // don't re-fetch the same matching rows (cartesian). We'll repeat them locally later.
        prevf = f; prevl = l;
        for (int r=0; r<l; r++) { // locate the corresponding matching row in right frame
          long loc = f+r-1;  // -1 because these are 0-based where 0 means no-match and 1 refers to the first row
          // TODO: could take / and % outside loop in cases where it doesn't span a batch boundary
          long row = _riteKO.at8order(loc); // right frame global row number that matches left frame
          // find the owning node for the row, using local operations here
          int chkIdx = _riteSB._vec.elem2ChunkIdx(row); //binary search in espc
          int ni = _riteSB._chunkNode[chkIdx];
          // TODO Split to an if() and batch and offset separately
          long pnl = perNodeRightLoc[ni]++;   // pnl = per node location.
          perNodeRightRows[ni][(int)(pnl/batchSizeLong)][(int)(pnl%batchSizeLong)] = row;  // ask that node for global row number row
        }
      }
    }
    // TODO assert that perNodeRite and Left are exactly equal to the number
    // expected and allocated.
    Arrays.fill(perNodeLeftLoc ,0); // clear for reuse below
    Arrays.fill(perNodeRightLoc,0); // denotes number of rows fetched 
  }

  private void chunksGetRawRemoteRows(final long perNodeLeftRows[][][], final long perNodeRightRows[][][], 
                                      GetRawRemoteRows grrrsLeft[][], GetRawRemoteRows grrrsRite[][]) {
    RPC<GetRawRemoteRows> grrrsRiteRPC[][] = new RPC[H2O.CLOUD.size()][];
    RPC<GetRawRemoteRows> grrrsLeftRPC[][] = new RPC[H2O.CLOUD.size()][];

    // Launch remote tasks left and right
    for( H2ONode node : H2O.CLOUD._memary ) {
      final int ni = node.index();
      final int bUppRite = perNodeRightRows[ni] == null ? 0 : perNodeRightRows[ni].length;
      final int bUppLeft =  perNodeLeftRows[ni] == null ? 0 :  perNodeLeftRows[ni].length;
      grrrsRiteRPC[ni] = new RPC[bUppRite];
      grrrsLeftRPC[ni] = new RPC[bUppLeft];
      grrrsRite[ni] = new GetRawRemoteRows[bUppRite];
      grrrsLeft[ni] = new GetRawRemoteRows[bUppLeft];
      for (int b = 0; b < bUppRite; b++) {
        // TODO try again now with better surrounding method
        // Arrays.sort(perNodeRightRows[ni][b]);  Simple quick test of fetching in monotonic order. Doesn't seem to help so far.
        grrrsRiteRPC[ni][b] = new RPC<>(node, new GetRawRemoteRows(_riteSB._frame, perNodeRightRows[ni][b])).call();
      }
      for (int b = 0; b < bUppLeft; b++) {
        // Arrays.sort(perNodeLeftRows[ni][b]);
        grrrsLeftRPC[ni][b] = new RPC<>(node, new GetRawRemoteRows(_leftSB._frame, perNodeLeftRows[ni][b])).call();
      }
    }
    for( H2ONode node : H2O.CLOUD._memary ) {
      // TODO: just send and wait for first batch on each node and then .get() next batch as needed.
      int ni = node.index();
      final int bUppRite = perNodeRightRows[ni] == null ? 0 : perNodeRightRows[ni].length;
      for (int b = 0; b < bUppRite; b++)
        _timings[5] += (grrrsRite[ni][b] = grrrsRiteRPC[ni][b].get()).timeTaken;
      final int bUppLeft = perNodeLeftRows[ni] == null ? 0 :  perNodeLeftRows[ni].length;
      for (int b = 0; b < bUppLeft; b++)
        _timings[5] += (grrrsLeft[ni][b] = grrrsLeftRPC[ni][b].get()).timeTaken;
    }
  }


  // Get Raw Remote Rows
  private void chunksGetRawRemoteRows(final long perNodeLeftRows[][][], GetRawRemoteRows grrrsLeft[][], int batchNumber) {
    RPC<GetRawRemoteRows> grrrsRiteRPC[][] = new RPC[H2O.CLOUD.size()][];
    RPC<GetRawRemoteRows> grrrsLeftRPC[][] = new RPC[H2O.CLOUD.size()][];

    // Launch remote tasks left and right
    for( H2ONode node : H2O.CLOUD._memary ) {
      final int ni = node.index();
      final int bUppLeft =  perNodeLeftRows[ni] == null ? 0 :  perNodeLeftRows[ni].length;  // denote nbatch
      grrrsLeftRPC[ni] = new RPC[bUppLeft];
      grrrsLeft[ni] = new GetRawRemoteRows[bUppLeft];
      if (batchNumber < bUppLeft) {
        grrrsLeftRPC[ni][batchNumber] = new RPC<>(node, new GetRawRemoteRows(_leftSB._frame, perNodeLeftRows[ni][batchNumber])).call();
      }
    }
    for( H2ONode node : H2O.CLOUD._memary ) {
      // TODO: just send and wait for first batch on each node and then .get() next batch as needed.
      int ni = node.index();
      final int bUppLeft = perNodeLeftRows[ni] == null ? 0 :  perNodeLeftRows[ni].length;
      if (batchNumber < bUppLeft)
        _timings[5] += (grrrsLeft[ni][batchNumber] = grrrsLeftRPC[ni][batchNumber].get()).timeTaken;
    }
  }

  // Now loop through _ret1st and _retLen and populate
  private void chunksPopulateRetFirst(final long perNodeLeftRows[][][], 
                                      long[] resultLeftLocPrevlPrevf, final int jb, final int numColsInResult,
                                      final int numLeftCols, final long perNodeLeftLoc[],
                                      final GetRawRemoteRows grrrsLeft[][], final double[][][] frameLikeChunks,
                                      BufferedString[][][] frameLikeChunks4String, final long[][][] frameLikeChunksLong) {
    // 16 bytes for each UUID (biggest type). Enum will be long (8).
    // TODO: How is non-Enum 'string' handled by H2O?
    final int batchSizeUUID = _retBatchSize;  // number of rows per chunk to fit in 256GB DKV limit.
    // TODO: hop back to original order here for [] syntax.
    long resultLoc = resultLeftLocPrevlPrevf[0];
    long leftLoc = resultLeftLocPrevlPrevf[1];
    long prevl = resultLeftLocPrevlPrevf[2];
    long prevf = resultLeftLocPrevlPrevf[3];

    if (jb < _ret1st.length) {
      for (int jo=0; jo<_ret1st[jb].length; ++jo) {        // jo = j offset
        leftLoc++;  // to save jb*_ret1st[0].length + jo;
        long f = _ret1st[jb][jo];  // TODO: take _ret1st[jb] outside inner loop
        long l = _retLen[jb][jo];
        if (f==0 && !_allLeft) continue;  // f==0 => left row matches to no right row
        // else insert the left row once and NA for the right columns i.e. left outer join

        // Fetch the left rows and recycle it if more than 1 row in the right table is matched to.
        // TODO could loop through batches rather than / and % wastefully
        long row = _leftKO.at8order(leftLoc);
        // TODO should leftOrder and retFirst/retLen have the same batch size to make this easier?
        // TODO Can we not just loop through _leftKO._order only? Why jb and jo too through
        int chkIdx = _leftSB._vec.elem2ChunkIdx(row); //binary search in espc
        int ni = _leftSB._chunkNode[chkIdx];
        long pnl = perNodeLeftLoc[ni]++;   // pnl = per node location.  TODO: batch increment this rather than
        int b = (int)(pnl / batchSizeUUID);  // however, the batch number of remote nodes may not match with final batch number
        int o = (int)(pnl % batchSizeUUID);
        if (grrrsLeft[ni]==null || grrrsLeft[ni][b] == null || grrrsLeft[ni][b]._chk==null) { // fetch chunk from remote nodes
          chunksGetRawRemoteRows(perNodeLeftRows, grrrsLeft, b);
        }
        long[][] chksLong= grrrsLeft[ni][b]._chkLong;
        double[][] chks = grrrsLeft[ni][b]._chk;
        BufferedString[][] chksString = grrrsLeft[ni][b]._chkString;

        final int l1 = Math.max((int)l,1);
        for (int rep = 0; rep < l1; rep++) {
          long a = resultLoc + rep;
          // TODO: loop into batches to save / and % for each repeat and still
          // cater for crossing multiple batch boundaries
          int whichChunk = (int) (a / batchSizeUUID);  // this actually points to batch number
          int offset = (int) (a % batchSizeUUID);

          for (int col=0; col<chks.length; col++) { // copy over left frame to frameLikeChunks
            if (this._stringCols[col]) {
              frameLikeChunks4String[col][whichChunk][offset] = chksString[col][o];
            } else if (this._intCols[col]) {
              frameLikeChunksLong[col][whichChunk][offset] = chksLong[col][o];
            } else {
                frameLikeChunks[col][whichChunk][offset] = chks[col][o];  // colForBatch.atd(row);
            }
          }
        }
        if (f==0) { resultLoc++; continue; } // no match so just one row (NA for right table) to advance over
        assert l > 0;
        if (prevf == f && prevl == l) {
          // just copy from previous batch in the result (populated by for()
          // below).  Contiguous easy in-cache copy (other than batches).
          for (int r=0; r<l; r++) {
            // TODO: loop into batches to save / and % for each repeat and
            // still cater for crossing multiple batch boundaries
            int toChunk = (int) (resultLoc / batchSizeUUID);  
            int toOffset = (int) (resultLoc % batchSizeUUID);
            int fromChunk = (int) ((resultLoc - l) / batchSizeUUID);
            int fromOffset = (int) ((resultLoc - l) % batchSizeUUID);
            for (int col=0; col<numColsInResult-numLeftCols; col++) {
              int colIndex = numLeftCols + col;
              if (this._stringCols[colIndex]) {
                frameLikeChunks4String[colIndex][toChunk][toOffset] = frameLikeChunks4String[colIndex][fromChunk][fromOffset];
              } else if (this._intCols[colIndex]) {
                frameLikeChunksLong[colIndex][toChunk][toOffset] = frameLikeChunksLong[colIndex][fromChunk][fromOffset];
              } else {
                frameLikeChunks[colIndex][toChunk][toOffset] = frameLikeChunks[colIndex][fromChunk][fromOffset];
              }
            }
            resultLoc++;
          }
          continue;
        }
        prevf = f;
        prevl = l;
      }
    }
    resultLeftLocPrevlPrevf[0] = resultLoc;
    resultLeftLocPrevlPrevf[1] = leftLoc;
    resultLeftLocPrevlPrevf[2] = prevl;
    resultLeftLocPrevlPrevf[3] = prevf;
  }

  // Now loop through _ret1st and _retLen and populate
  private void chunksPopulateRetFirst(final int batchSizeUUID, final int numColsInResult, final int numLeftCols, 
                                      final long perNodeLeftLoc[], final GetRawRemoteRows grrrsLeft[][], 
                                      final long perNodeRightLoc[], final GetRawRemoteRows grrrsRite[][], 
                                      final double[][][] frameLikeChunks, BufferedString[][][] frameLikeChunks4String, 
                                      final long[][][] frameLikeChunksLong) {
    // 16 bytes for each UUID (biggest type). Enum will be long (8).
    // TODO: How is non-Enum 'string' handled by H2O?

    long resultLoc=0;   // sweep upwards through the final result, filling it in
    // TODO: hop back to original order here for [] syntax.
    long leftLoc=_leftFrom; // sweep through left table along the sorted row locations.
    long prevf = -1, prevl = -1;
    for (int jb=0; jb<_ret1st.length; ++jb) {              // jb = j batch
      for (int jo=0; jo<_ret1st[jb].length; ++jo) {        // jo = j offset
        leftLoc++;  // to save jb*_ret1st[0].length + jo;
        long f = _ret1st[jb][jo];  // TODO: take _ret1st[jb] outside inner loop
        long l = _retLen[jb][jo];
        if (f==0 && !_allLeft) continue;  // f==0 => left row matches to no right row
        // else insert the left row once and NA for the right columns i.e. left outer join

        // Fetch the left rows and recycle it if more than 1 row in the right table is matched to.
        // TODO could loop through batches rather than / and % wastefully
        long row = _leftKO.at8order(leftLoc);
        // TODO should leftOrder and retFirst/retLen have the same batch size to make this easier?
        // TODO Can we not just loop through _leftKO._order only? Why jb and jo too through
        int chkIdx = _leftSB._vec.elem2ChunkIdx(row); //binary search in espc
        int ni = _leftSB._chunkNode[chkIdx];
        long pnl = perNodeLeftLoc[ni]++;   // pnl = per node location.  TODO: batch increment this rather than
        int b = (int)(pnl / batchSizeUUID);
        int o = (int)(pnl % batchSizeUUID);
        long[][] chksLong= grrrsLeft[ni][b]._chkLong;
        double[][] chks = grrrsLeft[ni][b]._chk;
        BufferedString[][] chksString = grrrsLeft[ni][b]._chkString;

        final int l1 = Math.max((int)l,1);
        for (int rep = 0; rep < l1; rep++) {
          long a = resultLoc + rep;
          // TODO: loop into batches to save / and % for each repeat and still
          // cater for crossing multiple batch boundaries
          int whichChunk = (int) (a / batchSizeUUID);
          int offset = (int) (a % batchSizeUUID);

          for (int col=0; col<chks.length; col++) { // copy over left frame to frameLikeChunks
            if (this._stringCols[col]) {
              if (chksString[col][o] != null)
                frameLikeChunks4String[col][whichChunk][offset] = chksString[col][o];
            } else if( _intCols[col] ) {
              frameLikeChunksLong[col][whichChunk][offset] = chksLong[col][o];
            } else
              frameLikeChunks[col][whichChunk][offset] = chks[col][o];  // colForBatch.atd(row);
          }
        }
        if (f==0) { resultLoc++; continue; } // no match so just one row (NA for right table) to advance over
        assert l > 0;
        if (prevf == f && prevl == l) {
          // just copy from previous batch in the result (populated by for()
          // below).  Contiguous easy in-cache copy (other than batches).
          for (int r=0; r<l; r++) {
            // TODO: loop into batches to save / and % for each repeat and
            // still cater for crossing multiple batch boundaries
            int toChunk = (int) (resultLoc / batchSizeUUID);
            int toOffset = (int) (resultLoc % batchSizeUUID);
            int fromChunk = (int) ((resultLoc - l) / batchSizeUUID);
            int fromOffset = (int) ((resultLoc - l) % batchSizeUUID);
            for (int col=0; col<numColsInResult-numLeftCols; col++) {
              int colIndex = numLeftCols + col;
              if (this._stringCols[colIndex]) {
                frameLikeChunks4String[colIndex][toChunk][toOffset] = frameLikeChunks4String[colIndex][fromChunk][fromOffset];
              } else if( _intCols[colIndex] ) {
                frameLikeChunksLong[colIndex][toChunk][toOffset] = frameLikeChunksLong[colIndex][fromChunk][fromOffset];
              } else {
                frameLikeChunks[colIndex][toChunk][toOffset] = frameLikeChunks[colIndex][fromChunk][fromOffset];
              }
            }
            resultLoc++;
          }
          continue;
        }
        prevf = f;
        prevl = l;
        for (int r=0; r<l; r++) {
          // TODO: loop into batches to save / and % for each repeat and still
          // cater for crossing multiple batch boundaries
          int whichChunk = (int) (resultLoc / batchSizeUUID);
          int offset = (int) (resultLoc % batchSizeUUID);
          long loc = f+r-1;  // -1 because these are 0-based where 0 means no-match and 1 refers to the first row
          // TODO: could take / and % outside loop in cases where it doesn't span a batch boundary
          row = _riteKO.at8order(loc);
          // find the owning node for the row, using local operations here
          chkIdx = _riteSB._vec.elem2ChunkIdx(row); //binary search in espc
          ni = _riteSB._chunkNode[chkIdx];
          pnl = perNodeRightLoc[ni]++;   // pnl = per node location.   // TODO Split to an if() and batch and offset separately
          chks = grrrsRite[ni][(int)(pnl / batchSizeUUID)]._chk;
          chksLong = grrrsRite[ni][(int)(pnl / batchSizeUUID)]._chkLong;
          chksString = grrrsRite[ni][(int)(pnl / batchSizeUUID)]._chkString;
          o = (int)(pnl % batchSizeUUID);
          for (int col=0; col<numColsInResult-numLeftCols; col++) {
            // TODO: this only works for numeric columns (not for UUID, strings, etc.)
            int colIndex = numLeftCols + col;
            if (this._stringCols[colIndex]) {
              if (chksString[_numJoinCols + col][o]!=null)
                frameLikeChunks4String[colIndex][whichChunk][offset] = chksString[_numJoinCols + col][o];  // colForBatch.atd(row);
            } else if( _intCols[colIndex] ) {
              frameLikeChunksLong[colIndex][whichChunk][offset] = chksLong[_numJoinCols + col][o];
            } else
              frameLikeChunks[colIndex][whichChunk][offset] = chks[_numJoinCols + col][o];  // colForBatch.atd(row);
          }
          resultLoc++;
        }
      }
    }
  }


  private void cleanUpMemory(GetRawRemoteRows[][] grrr, int batchIdx) {
    if (grrr != null) {
      int nodeNum = grrr.length;
      for (int nodeIdx = 0; nodeIdx < nodeNum; nodeIdx++) {
        int batchLimit = Math.min(batchIdx + 1, grrr[nodeIdx].length);
        if ((grrr[nodeIdx] != null) && (grrr[nodeIdx].length > 0)) {
          for (int bIdx = 0; bIdx < batchLimit; bIdx++) { // clean up memory
            int chkLen = grrr[nodeIdx][bIdx] == null ? 0 :
                    (grrr[nodeIdx][bIdx]._chk == null ? 0 : grrr[nodeIdx][bIdx]._chk.length);
            for (int cindex = 0; cindex < chkLen; cindex++) {
              grrr[nodeIdx][bIdx]._chk[cindex] = null;
              grrr[nodeIdx][bIdx]._chkString[cindex] = null;
              grrr[nodeIdx][bIdx]._chkLong[cindex] = null;
            }
            if (chkLen > 0) {
              grrr[nodeIdx][bIdx]._chk = null;
              grrr[nodeIdx][bIdx]._chkString = null;
              grrr[nodeIdx][bIdx]._chkLong = null;
            }
          }
        }
      }
    }
  }

  // compress all chunks and store them
  private void chunksCompressAndStore(final int b, final int numColsInResult, final double[][][] frameLikeChunks, 
                                      BufferedString[][][] frameLikeChunks4String, final long[][][] frameLikeChunksLong) {
    // compress all chunks and store them
    Futures fs = new Futures();
    for (int col = 0; col < numColsInResult; col++) {
      if (this._stringCols[col]) {
          NewChunk nc = new NewChunk(null, 0);
          for (int index = 0; index < frameLikeChunks4String[col][b].length; index++)
            nc.addStr(frameLikeChunks4String[col][b][index]);
          Chunk ck = nc.compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunks4String[col][b] = null; //free mem as early as possible (it's now in the store)
      } else if( _intCols[col] ) {
          NewChunk nc = new NewChunk(null,-1);
          for(long l: frameLikeChunksLong[col][b]) {
            if( l==Long.MIN_VALUE ) nc.addNA();
            else                    nc.addNum(l, 0);
          }
          Chunk ck = nc.compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunksLong[col][b] = null; //free mem as early as possible (it's now in the store)
      } else {
          Chunk ck = new NewChunk(frameLikeChunks[col][b]).compress();
          DKV.put(getKeyForMSBComboPerCol(_leftSB._msb, _riteSB._msb, col, b), ck, fs, true);
          frameLikeChunks[col][b] = null; //free mem as early as possible (it's now in the store)
      }
    }
    fs.blockForPending();
  }


  static Key getKeyForMSBComboPerCol(/*Frame leftFrame, Frame rightFrame,*/ int leftMSB, int rightMSB, int col /*final table*/, int batch) {
    return Key.make("__binary_merge__Chunk_for_col" + col + "_batch" + batch
        // + rightFrame._key.toString() + "_joined_with" + leftFrame._key.toString()
        + "_leftSB._msb" + leftMSB + "_riteSB._msb" + rightMSB,
      (byte) 1, Key.HIDDEN_USER_KEY, false, SplitByMSBLocal.ownerOfMSB(rightMSB==-1 ? leftMSB : rightMSB)
    ); //TODO home locally
  }

  static class GetRawRemoteRows extends DTask<GetRawRemoteRows> {
    Frame _fr;
    long[/*rows*/] _rows; //which rows to fetch from remote node, non-null on the way to remote, null on the way back

    double[/*col*/][] _chk; //null on the way to remote node, non-null on the way back
    BufferedString[][] _chkString;
    long[/*col*/][] _chkLong;

    double timeTaken;
    GetRawRemoteRows(Frame fr, long[] rows) { _rows = rows;  _fr = fr; }

    private static long[][] malloc8A(int m, int n) {
      long [][] res = new long[m][];
      for(int i = 0; i < m; ++i)
        res[i] = MemoryManager.malloc8(n);
      return res;
    }

    @Override
    public void compute2() {
      assert(_rows!=null);
      assert(_chk ==null);
      long t0 = System.nanoTime();
      _chk  = MemoryManager.malloc8d(_fr.numCols(),_rows.length);  // TODO: should this be transposed in memory?
      _chkLong = malloc8A(_fr.numCols(), _rows.length);
      _chkString = new BufferedString[_fr.numCols()][_rows.length];
      int cidx[] = MemoryManager.malloc4(_rows.length);
      int offset[] = MemoryManager.malloc4(_rows.length);
      Vec anyVec = _fr.anyVec();  assert anyVec != null;
      for (int row=0; row<_rows.length; row++) {
        cidx[row] = anyVec.elem2ChunkIdx(_rows[row]);  // binary search of espc array.  TODO: sort input row numbers to avoid
        offset[row] = (int)(_rows[row] - anyVec.espc()[cidx[row]]);
      }
      Chunk c[] = new Chunk[anyVec.nChunks()];
      for (int col=0; col<_fr.numCols(); col++) {
        Vec v = _fr.vec(col);
        for (int i=0; i<c.length; i++) c[i] = v.chunkKey(i).home() ? v.chunkForChunkIdx(i) : null;  // grab a chunk here
        if (v.isString()) {
          for (int row = 0; row < _rows.length; row++) {  // copy string and numeric columns
            _chkString[col][row] = c[cidx[row]].atStr(new BufferedString(), offset[row]); // _chkString[col][row] store by reference here
          }
        } else if( v.isInt() ) {
          for (int row = 0; row < _rows.length; row++) {  // extract info from chunks to one place
            _chkLong[col][row] = (c[cidx[row]].isNA(offset[row])) ? Long.MIN_VALUE : c[cidx[row]].at8(offset[row]);
          }
        } else {
          for (int row = 0; row < _rows.length; row++) {  // extract info from chunks to one place
            _chk[col][row] = c[cidx[row]].atd(offset[row]);
          }
        }
      }

      // tell remote node to fill up Chunk[/*batch*/][/*rows*/]
      // perNodeRows[node] has perNodeRows[node].length batches of row numbers to fetch
      _rows=null;
      _fr=null;
      assert(_chk != null && _chkLong != null);

      timeTaken = (System.nanoTime() - t0) / 1e9;

      tryComplete();
    }
  }
}
