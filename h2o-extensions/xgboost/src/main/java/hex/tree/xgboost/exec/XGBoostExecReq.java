package hex.tree.xgboost.exec;

import java.io.Serializable;
import java.util.Map;

public class XGBoostExecReq implements Serializable {
    
    public static class Init extends XGBoostExecReq {
        public int num_nodes;
        public Map<String, Object> parms;
        public String matrix_dir_path;
        public String[] nodes;
        public byte[] checkpoint_bytes;
        public String featureMap;
    }

    public static class Update extends XGBoostExecReq {
        public int treeId;
    }

    public static class GetMatrix extends XGBoostExecReq {
        public String matrix_dir_path;
    }

}
