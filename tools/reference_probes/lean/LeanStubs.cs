using System;
using System.Collections.Generic;
using System.Linq;

namespace MathNet.Numerics.Distributions
{
    public sealed class Normal
    {
        public double CumulativeDistribution(double value)
        {
            // Only needed to compile the complete upstream source file.  The
            // integration probe calls Statistics.SharpeRatio and never this.
            return 0.5 * (1.0 + Math.Tanh(value));
        }
    }
}

namespace MathNet.Numerics.Statistics
{
    public static class StatisticsExtensions
    {
        public static double Variance(this IEnumerable<double> values)
        {
            var rows = values.ToArray();
            if (rows.Length < 2) return double.NaN;
            var mean = rows.Average();
            return rows.Sum(value => Math.Pow(value - mean, 2)) / (rows.Length - 1);
        }

        public static double StandardDeviation(this IEnumerable<double> values)
        {
            return Math.Sqrt(values.Variance());
        }

        public static double Skewness(this IEnumerable<double> values) { return 0.0; }
        public static double Kurtosis(this IEnumerable<double> values) { return 3.0; }
    }
}

namespace System
{
    public static class LeanCompatibilityExtensions
    {
        public static bool IsNaNOrInfinity(this double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value);
        }

        public static bool IsNaNOrZero(this double value)
        {
            return double.IsNaN(value) || value == 0.0;
        }

        public static decimal SafeDecimalCast(this double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) return 0m;
            return (decimal)value;
        }
    }
}

namespace QuantConnect.Logging
{
    public static class Log
    {
        public static void Error(Exception ignored) { }
    }
}

namespace QuantConnect.Statistics
{
    public sealed class DrawdownMetrics
    {
        public decimal MaxDrawdown { get; private set; }
        public int MaxRecoveryTime { get; private set; }

        public DrawdownMetrics(decimal maxDrawdown, int maxRecoveryTime)
        {
            MaxDrawdown = maxDrawdown;
            MaxRecoveryTime = maxRecoveryTime;
        }
    }
}
