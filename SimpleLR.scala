package com.deepak.machinelearning

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.linalg.Vectors


/**
  * Created by Deepak Singh on 04/12/18.
  * Problem Statement: Implementing a basic Model for Simple Linear Regression Algorithm
  */

object SimpleLR {
  
  def main(args: Array[String]): Unit = {
    
    // Suppress the Log
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    
    // Creating Spark Session object
    
    val ss = new SparkSession.Builder()
                             .appName("Simle Linear Regression")
                             .master("local[2]")
                             .config("spark.sql.warehouse.dir", "stmp/sparksql")
                             .getOrCreate()
   // Loading the data                          
         
    val inputData = ss.read
                      .option("header", true)
                      .option("inferSchema", true)
                      .format("csv")
                      .load("Data/SimpleLR.csv")
                      
     inputData.printSchema()
     
     // Preparing input data to apply Linear regression Algorithm
     // Creating label and features. Here Label is PRICE. feature = [] --> Vector
     // Vector is collection of one dimension hetrogenous object
     // you should pick only the relevant column which satisfy the feature.
     
     val dataDf =  inputData.select(col("Price").as("Label"), col("Area"))
     
     dataDf.printSchema()
     
     // Assembler is responsible for collecting the feature variables
     // SparkML provides VectorAssembler to achieve this task
     
     val assembler = new VectorAssembler().setInputCols(Array("Area"))
                                          .setOutputCol("features")
                                          
    // Transform the data to assembler 
                                          
      val finalData = assembler.transform(dataDf).select(col("label"), col("features"))
      
      finalData.printSchema()
      finalData.show()

   // Building a model for Linear Regression algorithm
      
      val lrModel = new LinearRegression()
    
    // Train my model to provide prediction
    // he will predict the price based on the area of house (sqft)
    
    val lrModelexperiance = lrModel.fit(finalData)
    
    val learned = lrModelexperiance.summary
    
     learned.predictions.show()
     
     // Residuals defines the performance b/w the actual and predicted label
     
     learned.residuals.show() //residuals is difference b/w label and prediction
     
     println("IMPROVISING !!!")
    
    // Creating a training and testing set for building a Model
     
     val Array(training, testing) = finalData.randomSplit(Array(0.70, 0.30))
     
     println("Training Data")
     
     training.show()
     
     println("Testing Data")
     
     testing.show()
     
     println("Creating a new Model")
     
     val modelTraining = lrModel.fit(training)
     
     modelTraining.summary.predictions.show
     
     println(" Testing the Model")
     
     val test = modelTraining.transform(testing)
     
     test.show()
     
     //Let's give 5000 as the feature and get the predicted price from the model
     
     val newData = ss.createDataFrame(
         
                    Seq(Vectors.dense(5000)).map(Tuple1.apply)
                    ).toDF("features")
                    
   newData.show()
   
   lrModelexperiance.transform(newData).show
   modelTraining.transform(newData).show()
     
                      
  }
}
