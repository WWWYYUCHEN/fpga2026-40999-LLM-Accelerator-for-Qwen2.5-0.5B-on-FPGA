$ErrorActionPreference = "Stop"

if (-not ("QwenSerialTerminal" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.IO.Ports;
using System.Text;
using System.Threading;

public static class QwenSerialTerminal
{
    public static void Run(string preferredPort)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        Console.InputEncoding = new UTF8Encoding(false);

        string[] ports = SerialPort.GetPortNames();
        if (ports.Length == 0)
            throw new InvalidOperationException("No COM port found. Check the KV260 USB cable and driver.");

        string selected = null;
        foreach (string name in ports)
        {
            if (string.Equals(name, preferredPort, StringComparison.OrdinalIgnoreCase))
            {
                selected = name;
                break;
            }
        }
        if (selected == null)
            selected = ports[0];

        SerialPort port = new SerialPort(selected, 115200, Parity.None, 8, StopBits.One);
        port.Handshake = Handshake.None;
        port.DtrEnable = false;
        port.RtsEnable = false;
        port.ReadTimeout = 500;
        port.WriteTimeout = 1000;

        object outputLock = new object();
        Decoder decoder = Encoding.UTF8.GetDecoder();
        port.DataReceived += delegate
        {
            try
            {
                int available = port.BytesToRead;
                if (available <= 0)
                    return;

                byte[] bytes = new byte[available];
                int received = port.Read(bytes, 0, bytes.Length);
                if (received <= 0)
                    return;

                char[] chars = new char[Encoding.UTF8.GetMaxCharCount(received)];
                int count = decoder.GetChars(bytes, 0, received, chars, 0, false);
                lock (outputLock)
                {
                    Console.Write(new string(chars, 0, count));
                }
            }
            catch
            {
            }
        };

        try
        {
            port.Open();
        }
        catch (UnauthorizedAccessException)
        {
            throw new InvalidOperationException(
                "Serial port " + selected + " is busy. Close every other serial terminal first.");
        }

        Console.Clear();
        Console.WriteLine("============================================================");
        Console.WriteLine(" KV260 Qwen2.5-0.5B interactive UART");
        Console.WriteLine(" Port: " + selected + " | 115200 8N1 | no flow control");
        Console.WriteLine(" Record the console window while testing.");
        Console.WriteLine(" Type /quit-local and press Enter to close this terminal.");
        Console.WriteLine("============================================================");
        Console.WriteLine();

        Thread.Sleep(300);
        port.Write("\r");

        while (true)
        {
            string line = Console.ReadLine();
            if (line == null)
                break;
            if (line.Trim() == "/quit-local")
                break;
            if (line == "/help-local")
            {
                Console.WriteLine("[local] Type Chinese/English and press Enter. /quit-local closes this window.");
                continue;
            }

            byte[] data = Encoding.UTF8.GetBytes(line + "\r\n");
            port.Write(data, 0, data.Length);
        }

        try
        {
            port.Close();
        }
        catch
        {
        }
    }
}
"@
}

$preferred = $env:KV260_COM
if ([string]::IsNullOrWhiteSpace($preferred)) {
    $preferred = "COM11"
}

[QwenSerialTerminal]::Run($preferred)
