import ml.MLException;
import ml.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BaggingEnsemble extends Ensemble {
    private Integer bagSize;

    public BaggingEnsemble(final Matrix features, final Matrix labels, int bagSize){
        super(features, labels);
        this.modelWeighter = new EqualWeighter();
        this.bagSize = bagSize;
    }

    @Override
    public void train(final Matrix features, final Matrix labels){
        if (models == null || models.isEmpty()){
            throw new MLException("Cannot train an empty ensemble");
        }
        this.features = features;
        this.labels = labels;
        for (Model model : models){
            List<Matrix> matrices = fillABag(bagSize);
            model.getLearner().train(matrices.get(0), matrices.get(1));
        }
    }

    /**
     * Randomly selects rows from this.features and this.labels with replacement, filling a new Matrix.
     * First element is features matrix, second element is labels matrix.
     * @return
     */
    private List<Matrix> fillABag(int numRows){
        List<Matrix> newMatrices = new ArrayList<Matrix>();

        Matrix newFeatures = new Matrix(this.features, true);
        Matrix newLabels = new Matrix(this.labels, true);

        Random random = new Random(System.currentTimeMillis());

        for (int i = 0; i < numRows; i++){
            int row = random.nextInt(this.features.getNumRows());
            newFeatures.addRow(this.features.getRow(row));
            newLabels.addRow(this.labels.getRow(row));
        }

        newMatrices.add(newFeatures);
        newMatrices.add(newLabels);

        return newMatrices;
    }
}
