package hex.schemas;

import hex.tree.xgboost.exec.XGBoostExecResp;
import org.apache.commons.codec.binary.Base64;
import water.Key;
import water.api.API;
import water.api.Schema;
import water.api.schemas3.KeyV3;

public class XGBoostExecRespV3 extends Schema<XGBoostExecResp, XGBoostExecRespV3> {

    public XGBoostExecRespV3() {}
    
    public XGBoostExecRespV3(Key key) {
        this.key = KeyV3.make(key);
    }

    @API(help="Identifier")
    public KeyV3 key;

}
