import helpers.Counter;
import ml.ColumnAttributes;
import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class Ensemble extends SupervisedLearner{
    protected Matrix features, labels;
    protected List<Model> models;

    protected ModelWeighter modelWeighter;

    public Ensemble(){
        models = new ArrayList<Model>();
    }

    public Ensemble(Matrix features, Matrix labels){
        this.features = features;
        this.labels = labels;
        this.models = new ArrayList<Model>();
    }

    public Ensemble(ModelWeighter modelWeighter){
        models = new ArrayList<Model>();
        this.modelWeighter = modelWeighter;
    }

    @Override
    public void train(final Matrix features, final Matrix labels) {
        if (models == null || models.isEmpty()){
            throw new MLException("Cannot train an empty ensemble");
        }
        this.features = features;
        this.labels = labels;
        for (Model model : models){
            model.getLearner().train(features, labels);
        }
    }

    /**
     * Only supports one label.
     * @param in
     * @return
     */
    @Override
    public List<Double> predict(List<Double> in) {
        if(this.features ==null || this.labels == null){
            throw new MLException("Cannot predict without features and labels");
        }
        List<Double> labelPredictions = new ArrayList<Double>();
        for (Model model : models){
            labelPredictions.add(model.getLearner().predict(in).get(0));
        }

        List<Double> prediction = new ArrayList<Double>();
        prediction.add(baselineValue(labelPredictions));
        return prediction;
    }

    public void addModel(SupervisedLearner model){
        this.models.add(new Model(model));
    }

    public void addModel(Model model){
        this.models.add(model);
    }

    public void removeModel(SupervisedLearner model){
        this.models.remove(model);
    }

    public void removeModel(Model model){
        this.models.remove(model);
    }

    private Double baselineValue(List<Double> values){
        if(this.labels.getColumnAttributes(0).getColumnType()==ColumnAttributes.ColumnType.CATEGORICAL){ //calculate mode
            Counter<Double> counts = new Counter<Double>();

            for(Double value : values){
                counts.increment(value);
            }

            return counts.getMax();
        }
        else { //calculate mean
            int count = 0;
            Double v = 0.0;

            for (Double value: values){
                count++;
                v += value;
            }
            return v/count;
        }
    }

    //GETTERS AND SETTERS
    public ModelWeighter getModelWeighter() {
        return modelWeighter;
    }

    public void setModelWeighter(ModelWeighter modelWeighter) {
        this.modelWeighter = modelWeighter;
    }

    public Matrix getFeatures() {
        return features;
    }

    public void setFeatures(Matrix features) {
        this.features = features;
    }

    public Matrix getLabels() {
        return labels;
    }

    public void setLabels(Matrix labels) {
        this.labels = labels;
    }

    public List<Model> getModels() {
        return models;
    }

    public void setModels(List<Model> models) {
        this.models = models;
    }
}
