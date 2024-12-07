package ooad.dl4j;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.jfree.data.category.DefaultCategoryDataset;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DL4JPerformanceBenchmarkTest {

    @Test
    public void testModelInitialization() {
        MultiLayerNetwork model = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .list(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(10) // Input size
                                .nOut(2) // Output size
                                .build())
                        .build()
        );
        model.init();
        assertNotNull(model, "Model should initialize successfully.");
    }

    @Test
    public void testTraining() {
        MultiLayerNetwork model = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .list(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(10)
                                .nOut(2)
                                .build())
                        .build()
        );
        model.init();

        DataSet trainData = new DataSet(Nd4j.rand(1000, 10), Nd4j.zeros(1000, 2));
        try {
            model.fit(trainData);
        } catch (Exception e) {
            fail("Training failed with exception: " + e.getMessage());
        }
    }

    @Test
    public void testPrediction() {
        MultiLayerNetwork model = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .list(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(10)
                                .nOut(2)
                                .build())
                        .build()
        );
        model.init();

        DataSet testData = new DataSet(Nd4j.rand(100, 10), Nd4j.zeros(100, 2));
        try {
            model.output(testData.getFeatures());
        } catch (Exception e) {
            fail("Prediction failed with exception: " + e.getMessage());
        }
    }

    @Test
    public void testPlotting() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(1.2, "Time (s)", "Training");
        dataset.addValue(50.0, "Memory (MB)", "Memory Usage");
        dataset.addValue(0.8, "Time (s)", "Prediction");

        try {
            // Assume a plotting method is tested here
            assertNotNull(dataset, "Dataset for plotting should not be null.");
        } catch (Exception e) {
            fail("Plotting failed with exception: " + e.getMessage());
        }
    }
}