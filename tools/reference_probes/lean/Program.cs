using System;
using System.Globalization;
using QuantConnect.Statistics;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length != 3) return 2;
        var average = double.Parse(args[0], CultureInfo.InvariantCulture);
        var deviation = double.Parse(args[1], CultureInfo.InvariantCulture);
        var riskFree = double.Parse(args[2], CultureInfo.InvariantCulture);
        var value = Statistics.SharpeRatio(average, deviation, riskFree);
        Console.WriteLine(
            "{\"sharpe_ratio\":" + value.ToString("R", CultureInfo.InvariantCulture) + "}"
        );
        return 0;
    }
}
