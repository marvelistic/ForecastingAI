
using Microsoft.ML;
using Microsoft.ML.Data;

namespace StockPriceForecasting
{
    class Program
    {
        public class StockData
        {
            [LoadColumn(0)]
            public string Date { get; set; }
            [LoadColumn(1)]
            public float Open { get; set; }
            [LoadColumn(2)]
            public float High { get; set; }
            [LoadColumn(3)]
            public float Low { get; set; }
            [LoadColumn(4)]
            public float Close { get; set; }
        }

        public class StockPrediction
        {
            [ColumnName("Score")]
            public float PredictedClose { get; set; }

        }
        static void Main(string[] args)
        {
            //create ML context
            var mlContext = new MLContext();
            string path = @"D:\AI C#\ForecastingAI\ForecastingAI\stock_data.csv";
            Console.WriteLine(File.Exists(path));


            //load data
            var dataView = mlContext.Data.LoadFromTextFile<StockData>(
    @"D:\AI C#\ForecastingAI\ForecastingAI\stock_data.csv",
    separatorChar: ',',
    hasHeader: true
);
            //var dataView = mlContext.Data.LoadFromTextFile<StockData>("stock_data.csv", separatorChar: ',');
            var preview = dataView.Preview();
            foreach (var row in preview.RowView)
            {

                Console.Write($"{row.Values[0]} | {row.Values[1]} ");

            }
            var pipline = mlContext.Transforms.Concatenate("Features", nameof(StockData.Open), nameof(StockData.High), nameof(StockData.Low))
                            .Append(mlContext.Transforms.CopyColumns("Label", nameof(StockData.Close)))
                            .Append(mlContext.Regression.Trainers.FastTree());
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var model = pipline.Fit(trainTestData.TrainSet);
            var predictions = model.Transform(trainTestData.TrainSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            var predictionResults = mlContext.Data.CreateEnumerable<StockPrediction>(predictions, reuseRowObject: false);
            var testData = mlContext.Data.CreateEnumerable<StockData>(trainTestData.TestSet, reuseRowObject: false);
            foreach(var (predicted, actual ) in predictionResults.Zip(testData, (p,a) => (p, a)))
            {
                Console.Write($"Actual close price: {actual.Close}, Predicted close price:{predicted.PredictedClose}");
            }
        }

    }
}