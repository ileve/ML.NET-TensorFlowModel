using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using Microsoft.ML.Transforms;
using System.Linq;

namespace TestDeleteMe
{
    class Program
    {
        static string dataset = Path.Combine(Directory.GetCurrentDirectory(), "mpg.txt");
        static string pbModel = Path.Combine(Directory.GetCurrentDirectory(), "fromh5.pb");

        static void Main(string[] args)
        {
            var mlContext = new MLContext(); 

            string dataFile = File.ReadAllText(dataset);
            // Creating a data reader, based on the format of the data
            var reader = mlContext.Data.CreateTextLoader(new[] {
                new TextLoader.Column("dense_input_1", DataKind.Single, new[] {new TextLoader.Range(1,9)}),
                //new TextLoader.Column("index", DataKind.Int16, 0)
            },
            //ImagePath: c.LoadText(1)),
            separatorChar: '\t', hasHeader: true);

            //load tensorflow model
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(pbModel);
            var schema = tensorFlowModel.GetModelSchema();
            var inputSchema = tensorFlowModel.GetInputSchema();

            
            // Read the data
            var data = reader.Load(dataset);
            var estimator = tensorFlowModel.ScoreTensorFlowModel("dense_2_1/BiasAdd", "dense_input_1").Fit(data);

            //first way
            var engine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(estimator/*, inputSchema*/);

            var inputs = mlContext.Data.CreateEnumerable<InputData>(data, reuseRowObject: false).ToArray();

            for (int i = 0; i < inputs.Length; i++)
            {
                var predictedLabel = engine.Predict(inputs[i]);
                for (int j = 0; j < inputs[i].Features.Length; j++)
                {
                    Console.Write(inputs[i].Features[j]);
                    Console.Write(" ");
                }
                Console.WriteLine(predictedLabel.Output[0]);
            }


            ////second way
            //var transformedValues = estimator.Transform(data);
            //var outputs = mlContext.Data.CreateEnumerable<OutputData>(transformedValues, reuseRowObject: false);
            //foreach(var prediction in outputs)
            //{
            //    Console.WriteLine(prediction.Output[0]);
            //}
        }
    }

    class InputData
    {
        [ColumnName("dense_input_1"), VectorType(9)]
        public float[] Features { get; set; }
    }

    class OutputData
    {
        [ColumnName("dense_2_1/BiasAdd"), VectorType(1)]
        public float[] Output { get; set; }
    }
}
