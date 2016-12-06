package neural_network;

/**
 * Neural network
 * Created by mvukosav on 5.12.2016..
 */
public class Main {

    private static float[] in1 = {0.1f, 0.1f, 0.9f};
    private static float[] in2 = {0.1f, 0.9f, 0.1f};
    private static float[] in3 = {0.9f, 0.1f, 0.1f};

    private static float[] out1 = {0.9f, 0.1f, 0.1f};
    private static float[] out2 = {0.1f, 0.1f, 0.9f};
    private static float[] out3 = {0.1f, 0.9f, 0.1f};

    private static final float TRAINING_RATE = 0.5f;
    private static final int ITERATIONS = 10000;

    public static void main(String[] args) {
        NeuralNetwork_2H neuralNetwork_2H = new NeuralNetwork_2H(TRAINING_RATE, 3, 3, 3, 3);
        neuralNetwork_2H.addTraining(in1, out1);
        neuralNetwork_2H.addTraining(in2, out2);
        neuralNetwork_2H.addTraining(in3, out3);
        neuralNetwork_2H.run(ITERATIONS);
    }
}
