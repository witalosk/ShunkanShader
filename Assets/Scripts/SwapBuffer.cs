using System;

public class SwapBuffer<T> : IDisposable
{
    public T Read { get; protected set; }
    public T Write { get; protected set; }

    private readonly Action<T> _disposeAction;

    public SwapBuffer(Func<T> generateFunc, Action<T> disposeAction)
    {
        Read = generateFunc();
        Write = generateFunc();
        _disposeAction = disposeAction;
    }

    public SwapBuffer(T read, T write, Action<T> disposeAction)
    {
        Read = read;
        Write = write;
        _disposeAction = disposeAction;
    }
        
    public void Dispose()
    {
        _disposeAction?.Invoke(Read);
        _disposeAction?.Invoke(Write);
        Read = default;
        Write = default;
    }
        
    public void Swap()
    {
        (Read, Write) = (Write, Read);
    }
}