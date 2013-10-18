import ml.SupervisedLearner;

public class Model {
    private SupervisedLearner learner;

    private Double weight;

    public Model(){};

    public Model(SupervisedLearner learner){
        this.learner = learner;
    }

    public Model(SupervisedLearner learner, Double weight){
        this.learner = learner;
        this.weight = weight;
    }

    //GETTERS AND SETTERS
    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public SupervisedLearner getLearner() {
        return learner;
    }

    public void setLearner(SupervisedLearner learner) {
        this.learner = learner;
    }
}
