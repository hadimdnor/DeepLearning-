package ai.certifai.latihan;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class weatherForecast {

    //Formatted Date	Precip Type	Temperature (C)	Apparent Temperature (C)	Humidity	Wind Speed (km/h)	Wind Bearing (degrees)	Visibility (km)	Loud Cover	Pressure (millibars)
    public static final int trainSamples = 242;
    public static final int validSamples = 30;
    public static final int testSamples = 30;

    public static final int miniBatchSize = 10;
    public static final int numLabelClasses = 4;

    public static void main(String[] args) throws IOException, InterruptedException {
        File baseDir = new ClassPathResource("weather").getFile();
        File featuresDir = new File(baseDir, "feature");
        File labelsDir= new File(baseDir, "label");

        //load training data
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, trainSamples - 1));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(1, ",");
        trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, trainSamples - 1));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        /*
		#### LAB STEP 1 #####
		Load the validation data and testing data
		400 samples for validation and testing separately.
        */
        //load validation data
        SequenceRecordReader validFeatures = new CSVSequenceRecordReader(1, ",");
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", trainSamples, trainSamples + validSamples - 1));

        SequenceRecordReader validLabels = new CSVSequenceRecordReader(1, ",");
        validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", trainSamples, trainSamples + validSamples - 1));

        DataSetIterator validData = new SequenceRecordReaderDataSetIterator(validFeatures, validLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        //load testing data
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", trainSamples + validSamples, trainSamples + validSamples + testSamples- 1));

        SequenceRecordReader testLabels = new CSVSequenceRecordReader(1, ",");
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", trainSamples + validSamples, trainSamples + validSamples + testSamples - 1));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        int numInputs = trainData.inputColumns();
        int numClasses = 4;
        int epochs = 10;
        int seedNumber = 123;
        double learningRate = 0.001;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .seed(seedNumber)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .l2(0.001)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictWeather")
                .addLayer("layer0", new LSTM.Builder()
                                .nIn(numInputs)
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),"trainFeatures")
                .layer("layer1", new LSTM.Builder()
                                .nIn(100)
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),"layer0")
                .addLayer("predictWeather", new RnnOutputLayer.Builder()
                                .nIn(100)
                                .nOut(numClasses)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "layer1")
                .backpropType(BackpropType.Standard)
                .build();


        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));

        //int evalStep = 5;

//        for(int i = 0; i < epochs; ++i) {
//            model.fit(trainData);
//
//            ROC roc = new ROC(100);
//            while (validData.hasNext())
//            {
//                DataSet batch = validData.next();
//                INDArray[] output = model.output(batch.getFeatures());
//                List<String> labels = validData.getLabels();
//                roc.evalTimeSeries(batch.getLabels(), output[0]);
//
//            }
//
//            System.out.println("EPOCH " + i + " VALID AUC: " + roc.calculateAUC());
//            validData.reset();
//
//                /*
//                #### LAB STEP 2 #####
//                Save the model
//                */
//
//                //File locationToSave = new File(new ClassPathResource("weather/").getFile().getAbsolutePath().toString() + "_" + i + ".zip");
//                //ModelSerializer.writeModel(model, locationToSave, false);
//                //System.out.println("Model at epoch " + i + " save at " + locationToSave.toString());
//
//        }
//
//        //ROC
//        /*
//        #### LAB STEP 3 #####
//        Evaluate the results
//        */
//        ROC roc = new ROC(100);
//
//        while (testData.hasNext())
//        {
//            DataSet batch = testData.next();
//            INDArray[] output = model.output(batch.getFeatures());
//            roc.evalTimeSeries(batch.getLabels(), output[0]);
//        }
//        System.out.println("***** ROC Test Evaluation *****");
//        System.out.println(roc.calculateAUC());
//
//
//        testData.reset();

        //Evaluation
        Evaluation eval = new Evaluation(numLabelClasses);

        System.out.println("***** Test Evaluation *****");

        while(testData.hasNext())
        {
            DataSet testDataSet = testData.next();
            INDArray[] predicted = model.output(testDataSet.getFeatures());
            INDArray labels = testDataSet.getLabels();

            eval.evalTimeSeries(labels, predicted[0], testDataSet.getLabelsMaskArray());
        }

        for(int i=0; i < epochs; i++) {
            model.fit(trainData);
            eval = model.evaluate(testData);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
            testData.reset();
            trainData.reset();
        }


        System.out.println(eval.confusionToString());
        System.out.println(eval.stats());
    }
}
