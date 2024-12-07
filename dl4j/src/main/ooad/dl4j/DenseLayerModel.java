package ooad.dl4j;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DenseLayerModel implements Model {
    private MultiLayerNetwork network;

    public DenseLayerModel(int inputSize, int outputSize) {
        System.out.println("Initializing Dense Layer Model with input size " + inputSize + " and output size " + outputSize);
        network = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .list(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(inputSize).nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build());
        network.init();
    }

    @Override
    public INDArray predict(INDArray input) {
        System.out.println("Predicting with Dense Layer Model...");
        return network.output(input);
    }

    @Override
    public void fit(DataSetIterator data) {
        System.out.println("Training Dense Layer Model...");
        network.fit(data);
        System.out.println("Training complete.");
    }

    public static void main(String[] args) {
        DenseLayerModel model = new DenseLayerModel(10, 2);

        // Generate example input data
        INDArray input = Nd4j.rand(10, 10);
        INDArray predictions = model.predict(input);

        System.out.println("Predictions:\n" + predictions);
    }
}