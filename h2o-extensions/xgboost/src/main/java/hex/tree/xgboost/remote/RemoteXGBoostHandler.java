package hex.tree.xgboost.remote;

import hex.genmodel.utils.IOUtils;
import hex.schemas.XGBoostExecReqV3;
import hex.schemas.XGBoostExecRespV3;
import hex.tree.xgboost.exec.LocalXGBoostExecutor;
import hex.tree.xgboost.exec.XGBoostExecReq;
import org.apache.commons.codec.binary.Base64;
import water.AutoBuffer;
import water.Key;
import water.api.Handler;
import water.api.StreamingSchema;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class RemoteXGBoostHandler extends Handler {
    
    private static final Map<Key, LocalXGBoostExecutor> REGISTRY = new HashMap<>();

    private XGBoostExecRespV3 makeResponse(LocalXGBoostExecutor exec) {
        return new XGBoostExecRespV3(exec.modelKey);
    }
    
    private StreamingSchema makeStreamingResponse(byte[] data) {
        return new StreamingSchema(os -> {
            try {
                IOUtils.copyStream(new ByteArrayInputStream(data), os);
            } catch (IOException e) {
                throw new RuntimeException("Failed writing data to response.", e);
            }
        });
    }
    
    private LocalXGBoostExecutor getExecutor(XGBoostExecReqV3 req) {
        return REGISTRY.get(req.key.key());
    }

    @SuppressWarnings("unused")
    public XGBoostExecRespV3 init(int ignored, XGBoostExecReqV3 req) {
        XGBoostExecReq.Init init = (XGBoostExecReq.Init) AutoBuffer.javaSerializeReadPojo(Base64.decodeBase64(req.data));
        LocalXGBoostExecutor exec = new LocalXGBoostExecutor(req.key.key(), init);
        REGISTRY.put(exec.modelKey, exec);
        return makeResponse(exec);
    }

    @SuppressWarnings("unused")
    public StreamingSchema setup(int ignored, XGBoostExecReqV3 req) {
        LocalXGBoostExecutor exec = getExecutor(req);
        byte[] booster = exec.setup();
        return makeStreamingResponse(booster);
    }

    @SuppressWarnings("unused")
    public XGBoostExecRespV3 update(int ignored, XGBoostExecReqV3 req) {
        LocalXGBoostExecutor exec = getExecutor(req);
        XGBoostExecReq.Update update = (XGBoostExecReq.Update) AutoBuffer.javaSerializeReadPojo(Base64.decodeBase64(req.data));
        exec.update(update.treeId);
        return makeResponse(exec);
    }

    @SuppressWarnings("unused")
    public StreamingSchema getBooster(int ignored, XGBoostExecReqV3 req) {
        LocalXGBoostExecutor exec = getExecutor(req);
        byte[] booster = exec.updateBooster();
        return makeStreamingResponse(booster);
    }

    @SuppressWarnings("unused")
    public StreamingSchema getFeatureMap(int ignored, XGBoostExecReqV3 req) {
        LocalXGBoostExecutor exec = getExecutor(req);
        Object res = exec.getFeatureScores();
        return makeStreamingResponse(AutoBuffer.javaSerializeWritePojo(res));
    }

    @SuppressWarnings("unused")
    public XGBoostExecRespV3 cleanup(int ignored, XGBoostExecReqV3 req) {
        LocalXGBoostExecutor exec = getExecutor(req);
        exec.cleanup();
        REGISTRY.remove(exec.modelKey);
        return makeResponse(exec);
    }

}
