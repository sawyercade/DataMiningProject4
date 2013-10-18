import ml.ARFFParser;
import ml.Filter;
import ml.Imputer;
import ml.Matrix;
import ml.projectthree.EntropyReducingDecisionTreeLearner;
import ml.projecttwo.SlowKnnLearner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Project4 {

    public static void main(String[] args) throws IOException {
        runBagging(args);
    }

    public static void runBagging(String[] args) throws IOException{
        int featuresStart = 0, featuresEnd = 12;
        int labelsStart = 12, labelsEnd = 13;

        int k = 1;

        Matrix points = ARFFParser.loadARFF(args[0]);
        Matrix features = points.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = points.subMatrixCols(labelsStart, labelsEnd);

        BaggingEnsemble baggingEnsemble = new BaggingEnsemble(features, labels, features.getNumRows());

        EntropyReducingDecisionTreeLearner entropyReducingDecisionTreeLearner = new EntropyReducingDecisionTreeLearner(k);
        baggingEnsemble.addModel(entropyReducingDecisionTreeLearner);

        SlowKnnLearner slowKnnLearner = new SlowKnnLearner(5);
        baggingEnsemble.addModel(slowKnnLearner);

        Imputer imputer = new Imputer();
        Filter filter = new Filter(baggingEnsemble, imputer, true);

        filter.train(features, labels);

        List<Double> in = new ArrayList<Double>();
        in.addAll(Arrays.asList(new Double[]{0.7842, 12.5, 6.07, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.9, 9.14}));
        List<Double> prediction = filter.predict(in);

        System.out.println("Prediction: " + prediction.get(0));
    }

    public static void runEntropy(String[] args) throws IOException{
        int featuresStart = 0, featuresEnd = 12;
        int labelsStart = 12, labelsEnd = 13;

        int k = 1;

        Matrix points = ARFFParser.loadARFF(args[0]);
        Matrix features = points.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = points.subMatrixCols(labelsStart, labelsEnd);

        EntropyReducingDecisionTreeLearner entropyReducingDecisionTreeLearner = new EntropyReducingDecisionTreeLearner(k);

        Imputer imputer = new Imputer();
        Filter filter = new Filter(entropyReducingDecisionTreeLearner, imputer, true);
        // Filter filter = new Filter(randomDecisionTreeLearner, imputer, true);

        filter.train(features, labels);

        List<Double> in = new ArrayList<Double>();
        in.addAll(Arrays.asList(new Double[]{0.7842, 12.5, 6.07, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.9, 9.14}));
        List<Double> prediction = filter.predict(in);

        System.out.println("Prediction: " + prediction.get(0));

        //System.out.print(randomDecisionTreeLearner.getTreeString());
        //System.out.print(entropyReducingDecisionTreeLearner.getTreeString());
    }
}
