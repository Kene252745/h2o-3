package hex.tree.xgboost.matrix;

import hex.tree.xgboost.exec.XGBoostExecReq;
import hex.tree.xgboost.exec.XGBoostHttpClient;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.apache.log4j.Logger;
import water.H2O;

import java.io.File;
import java.io.IOException;

public class RemoteMatrixLoader extends MatrixLoader {

    private static final Logger LOG = Logger.getLogger(RemoteMatrixLoader.class);

    private final String remoteDirectory;
    private final String remoteNode;

    public RemoteMatrixLoader(String remoteDirectory, String[] nodes) {
        this.remoteDirectory = remoteDirectory;
        this.remoteNode = nodes[H2O.SELF.index()];
    }

    @Override
    public DMatrix makeLocalMatrix() throws IOException, XGBoostError {
        assert remoteNode != null : "Should not be loading DMatrix on this node.";
        File tempFile = File.createTempFile("dmatrix", ".bin");
        XGBoostExecReq.GetMatrix req = new XGBoostExecReq.GetMatrix();
        req.matrix_dir_path = remoteDirectory;
        LOG.debug("Downloading matrix data into " + tempFile);
        XGBoostHttpClient http = new XGBoostHttpClient(remoteNode);
        http.postFile(null, "getMatrix", req, tempFile);
        LOG.info("Downloading of remote matrix finished. Loading into memory.");
        try {
            return new DMatrix(tempFile.getAbsolutePath());
        } finally {
            tempFile.delete();
        }
    }
}
