package hex.tree.xgboost.exec;

import hex.DataInfo;
import hex.genmodel.utils.IOUtils;
import hex.schemas.XGBoostExecReqV3;
import hex.schemas.XGBoostExecRespV3;
import hex.tree.xgboost.BoosterParms;
import hex.tree.xgboost.XGBoostModel;
import hex.tree.xgboost.XGBoostUtils;
import hex.tree.xgboost.util.FeatureScore;
import ml.dmlc.xgboost4j.java.FrameXGBoostMatrixFactory;
import ml.dmlc.xgboost4j.java.XGBoostMatrixFactory;
import ml.dmlc.xgboost4j.java.XGBoostSaveMatrixTask;
import ml.dmlc.xgboost4j.java.XGBoostSetupTask;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.apache.log4j.Logger;
import water.AutoBuffer;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import water.util.IcedHashMap;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Map;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.apache.http.HttpHeaders.CONTENT_TYPE;
import static water.util.HttpResponseStatus.OK;

public class RemoteXGBoostExecutor implements XGBoostExecutor {

    private static final Logger LOG = Logger.getLogger(RemoteXGBoostExecutor.class);

    public final String baseUri;
    public final Key modelKey;
    
    public RemoteXGBoostExecutor(String remoteUri, XGBoostModel model, Frame train, XGBoostModel.XGBoostParameters parms) {
        baseUri = remoteUri + "/3/XGBoostExecutor.";
        modelKey = model._key;
        XGBoostExecReq.Init req = new XGBoostExecReq.Init();
        XGBoostSetupTask.FrameNodes trainFrameNodes = XGBoostSetupTask.findFrameNodes(train);
        req.num_nodes = trainFrameNodes.getNumNodes();
        if (parms.hasCheckpoint()) {
            req.checkpoint_bytes = model.model_info()._boosterBytes;
        }
        DataInfo dataInfo = model.model_info().dataInfo();
        req.parms = new IcedHashMap<>();
        req.parms.putAll(XGBoostModel.createParamsMap(parms, model._output.nclasses(), dataInfo.coefNames()));
        model._output._native_parameters = BoosterParms.fromMap(req.parms).toTwoDimTable();
        XGBoostMatrixFactory f = new FrameXGBoostMatrixFactory(model, parms, trainFrameNodes);
        req.matrix_dir_path = H2O.ICE_ROOT.toString() + "/" + modelKey.toString();
        new XGBoostSaveMatrixTask(modelKey, req.matrix_dir_path, trainFrameNodes._nodes, f).run();
        req.featureMap = XGBoostUtils.createFeatureMap(model, train);
        XGBoostExecRespV3 resp = post(modelKey, "init", req);
        assert modelKey.equals(resp.key.key());
    }

    interface ResponseTransformer<T> {
        T transform(HttpEntity e) throws IOException;
    }

    private static final ResponseTransformer<byte[]> ByteArrayResponseTransformer = (e) -> {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        IOUtils.copyStream(e.getContent(), bos);
        return bos.toByteArray();
    };

    private static final ResponseTransformer<XGBoostExecRespV3> JsonResponseTransformer = (e) -> {
        String responseBody = EntityUtils.toString(e);
        XGBoostExecRespV3 resp = new XGBoostExecRespV3();
        resp.fillFromBody(responseBody);
        return resp;
    };

    private XGBoostExecRespV3 post(Key key, String method, XGBoostExecReq reqContent) {
        return post(key, method, reqContent, JsonResponseTransformer);
    }

    private <T> T post(Key key, String method, XGBoostExecReq reqContent, ResponseTransformer<T> transformer) {
        XGBoostExecReqV3 req = new XGBoostExecReqV3(key, reqContent);
        HttpPost httpReq = new HttpPost(baseUri + method);
        httpReq.setEntity(new StringEntity(req.toJsonString(), UTF_8));
        httpReq.setHeader(CONTENT_TYPE, ContentType.APPLICATION_JSON.getMimeType());
        LOG.info("Remote XGBoost: Request " + method + " " + reqContent);
        try (CloseableHttpClient client = HttpClientBuilder.create().build();
             CloseableHttpResponse response = client.execute(httpReq)) {
            if (response.getStatusLine().getStatusCode() != OK.getCode()) {
                throw new IllegalStateException("Unexpected response (status: " + response.getStatusLine() + ").");
            }
            LOG.info("Remote XGBoost: Response received " + response.getEntity().getContentLength() + " bytes.");
            return transformer.transform(response.getEntity());
        } catch (IOException e) {
            throw new RuntimeException("HTTP Request failed", e);
        }
    }
    
    @Override
    public byte[] setup() {
        XGBoostExecReq req = new XGBoostExecReq(); // no req params
        return post(modelKey, "setup", req, ByteArrayResponseTransformer);
    }

    @Override
    public void update(int treeId) {
        XGBoostExecReq.Update req = new XGBoostExecReq.Update();
        req.treeId = treeId;
        XGBoostExecRespV3 resp = post(modelKey, "update", req);
        assert resp.key.key().equals(modelKey);
    }

    @Override
    public byte[] updateBooster() {
        XGBoostExecReq req = new XGBoostExecReq(); // no req params
        return post(modelKey, "getBooster", req, ByteArrayResponseTransformer);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Map<String, FeatureScore> getFeatureScores() {
        XGBoostExecReq req = new XGBoostExecReq(); // no req params
        byte[] data = post(modelKey, "getFeatures", req, ByteArrayResponseTransformer);
        return (Map<String, FeatureScore>) AutoBuffer.javaSerializeReadPojo(data);
    }

    @Override
    public void cleanup() {
        XGBoostExecReq req = new XGBoostExecReq(); // no req params
        XGBoostExecRespV3 resp = post(modelKey, "cleanup", req);
        assert resp.key.key().equals(modelKey);
    }
}
