/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.api.project;
import weka.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author franky
 */
public class WekaApiProject {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        LoadData load = new LoadData();
        buildClassifier();
        predict();
        
    }
    public static void buildClassifier() throws Exception{
        
        //Load dataset
        DataSource source = new DataSource("/home/franky/Desktop/dataset.arff");
        Instances dataset = source.getDataSet();
        
        dataset.setClassIndex(dataset.numAttributes() -1);
        
        //Building classifiers
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(dataset);
        System.out.println(naiveBayes.getCapabilities().toString());
        
        J48 tree = new J48();
        tree.buildClassifier(dataset);
        System.out.println(tree.graph());
        
        Logistic logistic = new Logistic();
        logistic.buildClassifier(dataset);
        System.out.println(logistic.getCapabilities().toString());
    
    }
    public static void predict() throws Exception {
        
        System.out.println(" Predicting ");
        
        //Loading training dataset
        DataSource trainSource = new DataSource("/home/franky/Desktop/trainDataset.arff");
        Instances trainDataset = trainSource.getDataSet();
        trainDataset.setClassIndex(trainDataset.numAttributes() -1);
        
        //Building a model
        J48 tree = new J48();
        tree.buildClassifier(trainDataset);
        System.out.println(tree);
        
        //Loading testDataset
        DataSource testSource = new DataSource("/home/franky/Desktop/testDataset.arff");
        Instances testDataset = testSource.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes() -1);
//        testDataset.setClass(testDataset.attribute("plas"));
        
        //Making predictions
        System.out.println("****** Started Predictions ******* ");
        
        for (int i = 0; i < testDataset.numInstances(); i++){
            
            //testInstance value
            double actualValue = testDataset.instance(i).classValue();
            
            //getting the test instance
            Instance newInstance = testDataset.instance(i);
            
            //classifying the instance
            double prediction = tree.classifyInstance(newInstance);
            
            System.out.println(actualValue + " prediction value is " + prediction);
            
        }
        
    }
    
}
