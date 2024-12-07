package ooad.dl4j;

//This file has to be executed to get the performance anlaysis graph.
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;

public class DL4JBenchmark {
    public static void main(String[] args) {
        int inputSize = 10;
        int outputSize = 2;

        // Initialize the model
        MultiLayerNetwork model = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .nIn(inputSize)
                                .nOut(outputSize)
                                .build())
                        .build()
        );

        model.init();

        // Generate mock data
        DataSet trainData = new DataSet(Nd4j.rand(1000, inputSize), Nd4j.zeros(1000, outputSize));
        DataSet testData = new DataSet(Nd4j.rand(100, inputSize), Nd4j.zeros(100, outputSize));

        // Measure memory and CPU
        MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage beforeMemory = memoryMXBean.getHeapMemoryUsage();
        long startTime = System.nanoTime();

        model.fit(trainData);

        long endTime = System.nanoTime();
        MemoryUsage afterMemory = memoryMXBean.getHeapMemoryUsage();

        double trainingTime = (endTime - startTime) / 1e9;
        double memoryUsage = (afterMemory.getUsed() - beforeMemory.getUsed()) / (1024 * 1024);

        System.out.println("DL4J: Training Time: " + trainingTime + " seconds");
        System.out.println("DL4J: Memory Usage: " + memoryUsage + " MB");

        // Prediction
        startTime = System.nanoTime();
        model.output(testData.getFeatures());
        endTime = System.nanoTime();

        double predictionTime = (endTime - startTime) / 1e9;
        System.out.println("DL4J: Prediction Time: " + predictionTime + " seconds");

        // Create a bar chart for performance metrics
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(trainingTime, "Time (s)", "Training");
        dataset.addValue(memoryUsage, "Memory (MB)", "Memory Usage");
        dataset.addValue(predictionTime, "Time (s)", "Prediction");

        JFreeChart barChart = ChartFactory.createBarChart(
                "DL4J Performance Metrics",
                "Metric",
                "Value",
                dataset
        );

        // Display the chart
        JFrame frame = new JFrame("Performance Metrics");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        ChartPanel chartPanel = new ChartPanel(barChart);
        frame.add(chartPanel);
        frame.setVisible(true);
    }
}