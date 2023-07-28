# spindle
Spindle neurons enable whales to be deep thinkers.

## How to run
1. Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and extract `.gz` files into [`./data/`](./data/) directory
2. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it spindle bash
    ```
3. Training:
    ```bash
    zig build run -- ../data/
    ```
    Logs:
    ```
    Before training: loss=1.6351758242, acc=10.53%
    Epoch 0: elapsed_time=0:24, loss=0.0950836316, acc=89.35%
    Epoch 1: elapsed_time=0:47, loss=0.0602058768, acc=92.61%
    Epoch 2: elapsed_time=1:11, loss=0.0562265515, acc=93.25%
    Epoch 3: elapsed_time=1:35, loss=0.0533989370, acc=93.56%
    Epoch 4: elapsed_time=1:59, loss=0.0503539443, acc=94.18%
    Epoch 5: elapsed_time=2:23, loss=0.0506065302, acc=94.01%
    Epoch 6: elapsed_time=2:46, loss=0.0488770492, acc=94.37%
    Epoch 7: elapsed_time=3:10, loss=0.0465301722, acc=94.54%
    Epoch 8: elapsed_time=3:34, loss=0.0448890477, acc=94.69%
    Epoch 9: elapsed_time=3:58, loss=0.0450211503, acc=94.71%
    Epoch 10: elapsed_time=4:21, loss=0.0452360660, acc=94.56%
    Epoch 11: elapsed_time=4:45, loss=0.0468302444, acc=94.37%
    Epoch 12: elapsed_time=5:09, loss=0.0434075668, acc=94.85%
    Epoch 13: elapsed_time=5:33, loss=0.0441294163, acc=94.76%
    Epoch 14: elapsed_time=5:56, loss=0.0428987928, acc=94.98%
    ```

## References
- [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
- [DNNs from Scratch in Zig](https://monadmonkey.com/dnns-from-scratch-in-zig)
