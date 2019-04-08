package weka.api.project;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

import java.io.File;

public class LoadData {
    public static void main (String [] args) throws Exception {
        
    }
    LoadData() throws Exception{
        //Loading datasets
        
        DataSource source = new DataSource("/home/franky/weka datasets/datasets-UCI/UCI/diabetes.arff");
        Instances dataset = source.getDataSet();
        System.out.println(" Initial dataset");
        System.out.println(dataset.toSummaryString());
        
        //Filtering attributes
        // removing the first attribute
        String[] opts = new String[]{"-R","1"};
        Remove remove = new Remove();
        remove.setOptions(opts);
        remove.setInputFormat(dataset);
        
        //apply filters
        Instances newDataset = Filter.useFilter(dataset, remove);
        System.out.println(" Filtered dataset");
        System.out.println(newDataset.toSummaryString());
        
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newDataset);
        saver.setFile(new File("/home/franky/Desktop/dataset.arff"));
        saver.writeBatch();
       
    
    }
    
}
