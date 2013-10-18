import helpers.Ranker;
import ml.ColumnAttributes;
import ml.MLException;
import ml.Matrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BombingEnsemble extends Ensemble{
    private Integer n; //number of random weighting combinations to try

    public BombingEnsemble(int n){
        this.n = n;
        this.modelWeighter = new RandExpWeighter();
    }

    @Override
    public void train(Matrix features, Matrix labels){
        this.features = features;
        this.labels = labels;

        //split the data in half
        List<Matrix> splitMatrices = splitIntoTwo(features, labels);
        Matrix trainingFeatures = splitMatrices.get(0);
        Matrix trainingLabels = splitMatrices.get(1);
        Matrix predictingFeatures = splitMatrices.get(2);
        Matrix predictingLabels = splitMatrices.get(3);

        Map<List<Double>, Double> weightingSses = new HashMap<List<Double>, Double>(); //stores the weighting combinations and
        for(int i=0; i<n; i++){
            this.modelWeighter.weight(this.models); //assign a weight to all the models
            List<Double> weights = new ArrayList<Double>(); //stores the weights

            //train each model and store the weights
            for(Model model : this.models){
                model.getLearner().train(trainingFeatures, trainingLabels);
                weights.add(model.getWeight());
            }
            weightingSses.put(weights, this.getAccuracy(predictingFeatures, predictingLabels));
        }

        //find the best weighting combination from among all of the combinations tried
        List<Double> bestWeights = getMinSseWeightCombination(weightingSses);

        //assign the best weighting combination to the models
        int weightCounter = 0;
        for (Model model : this.models){
            model.setWeight(bestWeights.get(weightCounter));
        }
    }

    @Override
    public List<Double> predict (List<Double> in){
        List<Double> iFuckingHateThatThisHasToBeAListInsteadOfASingleGoddamnLabel = new ArrayList<Double>();

        //if the label is categorical
        if (this.labels.getColumnAttributes(0).getColumnType() == ColumnAttributes.ColumnType.CATEGORICAL){
            //find and return the most heavily predicted label
            Ranker<Double> ranker = new Ranker<Double>();
            for (Model model : this.models){
                ranker.increase(model.getLearner().predict(in).get(0), model.getWeight());
            }

            iFuckingHateThatThisHasToBeAListInsteadOfASingleGoddamnLabel.add(ranker.getMax());
            return iFuckingHateThatThisHasToBeAListInsteadOfASingleGoddamnLabel;
        }
        //if the label is continuous
        else {
            //find and return sum(prediction_i * weight_i)/number_of_models , which is the weighted mean
            Double sum = 0.0;
            for (Model model : this.models){
                sum +=  model.getLearner().predict(in).get(0) * model.getWeight();
            }
            iFuckingHateThatThisHasToBeAListInsteadOfASingleGoddamnLabel.add(sum/this.models.size());
            return iFuckingHateThatThisHasToBeAListInsteadOfASingleGoddamnLabel;
        }
    }

    /**
     * Splits these matrices in half. In the returned matrix, elements 0 and 1 are the first features and labels, 2 and
     * 3 are the second features and labels.
     * @param features
     * @param labels
     * @return
     */
    private List<Matrix> splitIntoTwo(Matrix features, Matrix labels){
        List<Matrix> matrices = new ArrayList<Matrix>();
        int half = features.getNumRows()/2;
        Matrix features1 = features.subMatrixRows(0, half);
        Matrix labels1 = labels.subMatrixRows(0, half);
        Matrix features2 = features.subMatrixRows(half, features.getNumRows());
        Matrix labels2 = labels.subMatrixRows(half, labels.getNumRows());

        matrices.add(features1);
        matrices.add(labels1);
        matrices.add(features2);
        matrices.add(labels2);

        return matrices;
    }

    private static List<Double> getMinSseWeightCombination(Map<List<Double>, Double> map){
        List<Double> minList = null;
        Double min = Double.valueOf(Double.POSITIVE_INFINITY);
        for (Map.Entry<List<Double>, Double> entry : map.entrySet()){
            if (min.compareTo(entry.getValue())>0){
                minList = entry.getKey();
                min = entry.getValue();
            }
        }
        return minList;
    }
}
