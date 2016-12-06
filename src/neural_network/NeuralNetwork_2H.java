package neural_network;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by mvukosav on 6.12.2016..
 */
public class NeuralNetwork_2H {

    private float TRAINING_RATE;

    private int numInputLayer;
    private int numHiddenLayer1;
    private int numHiddenLayer2;
    private int numOutputLayer;

    private float inputs[];
    private float hidden1[];
    private float hidden2[];
    private float outputs[];

    private float hidden1_errors[];
    private float hidden2_errors[];
    private float output_errors[];

    //    weight from input to hidden1 layer
    private float W_ih1[][];
    //    weight from hidden1 to hidden2 layer
    private float W_h1h2[][];
    //    weight from hidden2 to output layer
    private float W_h2o[][];

    private List inputTraining = new ArrayList<>();
    private List outputTraining = new ArrayList<>();


    public NeuralNetwork_2H(float TRAINING_RATE, int numInputLayer, int numHiddenLayer1, int numHiddenLayer2, int numOutputLayer) {
        this.TRAINING_RATE = TRAINING_RATE;
        this.numInputLayer = numInputLayer;
        this.numHiddenLayer1 = numHiddenLayer1;
        this.numHiddenLayer2 = numHiddenLayer2;
        this.numOutputLayer = numOutputLayer;
        inputs = new float[numInputLayer];
        hidden1 = new float[numHiddenLayer1];
        hidden2 = new float[numHiddenLayer2];
        outputs = new float[numOutputLayer];

        W_ih1 = new float[numInputLayer][numHiddenLayer1];
        W_h1h2 = new float[numHiddenLayer1][numHiddenLayer2];
        W_h2o = new float[numHiddenLayer2][numOutputLayer];

        output_errors = new float[numOutputLayer];
        hidden1_errors = new float[numHiddenLayer1];
        hidden2_errors = new float[numHiddenLayer2];

        randomizeWeight(numInputLayer, numHiddenLayer1, W_ih1);
        randomizeWeight(numHiddenLayer1, numHiddenLayer2, W_h1h2);
        randomizeWeight(numHiddenLayer2, numOutputLayer, W_h2o);
    }

    public void addTraining(float[] inputExample, float[] outputExample) {
        if (inputs.length < 1 || outputExample.length < 1) {
            throw new IllegalArgumentException("addTraining parameters array size must not be zero");
        }

        if (inputs.length != inputExample.length || outputs.length != outputExample.length) {
            throw new IllegalArgumentException("addTraining parameters array size is not equal as neural network data");
        }

        inputTraining.add(inputExample);
        outputTraining.add(outputExample);
    }

    public void run(int iterations) {
        float error = 0;
        for (int i = 0; i < iterations; i++) {
            error += train();
            if (i > 0 && (i % 100 == 0)) {
                error /= 100;
                System.out.println("Cycle: " + i + ", error is: " + error);
                error = 0;
            }
            System.out.println("Iteration: " + i + ", error is: " + error);
        }

        for (int i = 0; i < inputTraining.size(); i++) {
            printResults((float[]) inputTraining.get(i), (float[]) outputTraining.get(i));
        }
    }


    private float[] recall(float[] input) {
        //refresh inputs
        int inputLength = input.length;
        for (int i = 0; i < inputLength; i++) {
            inputs[i] = input[i];
        }

        forwardPass();

        float[] ret = new float[numOutputLayer];
        int outputLength = ret.length;
        for (int i = 0; i < outputLength; i++) {
            ret[i] = outputs[i];
        }
        return ret;
    }

    private void forwardPass() {
        resetData(hidden1);
        resetData(hidden2);
        resetData(outputs);

        //refresh first part
        for (int i = 0; i < numInputLayer; i++) {
            for (int j = 0; j < numHiddenLayer1; j++) {
                hidden1[j] += inputs[i] * W_ih1[i][j]; // first is where it goes, second is from where it comes
            }
        }

        //refresh middle part
        for (int i = 0; i < numHiddenLayer1; i++) {
            for (int j = 0; j < numHiddenLayer2; j++) {
                hidden2[j] += hidden1[i] * W_h1h2[i][j];
            }
        }
        //refresh output part
        for (int i = 0; i < numHiddenLayer2; i++) {
            for (int j = 0; j < numOutputLayer; j++) {
                outputs[j] += sigmoid(hidden2[i]) * W_h2o[i][j];
            }
        }
    }


    private void resetData(float[] data) {
        int length = data.length;
        for (int i = 0; i < length; i++) {
            data[i] = 0.0f;
        }
    }

    private float train() {
        return train(inputTraining, outputTraining);
    }

    private int currentExample = 0;

    private float train(List inputTraining, List outputTraining) {

        if (inputTraining.size() < 1 || outputTraining.size() < 1) {
            throw new IllegalArgumentException("inputTraining and outputTraining must not be empty");
        }

        float error = 0.0f;
        int num_cases = inputTraining.size();
        resetData(hidden1_errors);
        resetData(hidden2_errors);
        resetData(output_errors);

        //copy the input value
        for (int i = 0; i < numInputLayer; i++) {
            inputs[i] = ((float[]) inputTraining.get(currentExample))[i];
        }

        //copy the output value
        float[] outs = ((float[]) outputTraining.get(currentExample));

        //Perform a forward pass through the network
        forwardPass();

//        output
        for (int i = 0; i < numOutputLayer; i++) {
            output_errors[i] = (outs[i] - outputs[i]) * sigmoidDerived(outputs[i]);
        }
//      hidden 2
        for (int i = 0; i < numHiddenLayer2; i++) {
            resetData(hidden2_errors);
            for (int j = 0; j < numOutputLayer; j++) {
                hidden2_errors[i] += output_errors[j] * W_h2o[i][j];
            }
        }
//      hidden 1
        for (int i = 0; i < numHiddenLayer1; i++) {
            resetData(hidden1_errors);
            for (int j = 0; j < numHiddenLayer2; j++) {
                hidden1_errors[i] += hidden2_errors[j] * W_h1h2[i][j];
            }
        }

//      update hidden1_errors
        for (int i = 0; i < numHiddenLayer1; i++) {
            hidden1_errors[i] = hidden1_errors[i] * sigmoidDerived(hidden1[i]);
        }
//      update hidden2_errors
        for (int i = 0; i < numHiddenLayer2; i++) {
            hidden2_errors[i] = hidden2_errors[i] * sigmoidDerived(hidden2[i]);
        }

//       update the hidden2 to output weights
        for (int i = 0; i < numOutputLayer; i++) {
            for (int j = 0; j < numHiddenLayer2; j++) {
                float updatedValue = W_h2o[j][i] + TRAINING_RATE * output_errors[i] * hidden2[j];
                W_h2o[j][i] = clampWeight(updatedValue);
            }
        }

//       update the hidden1 to hidden2 weights
        for (int i = 0; i < numHiddenLayer2; i++) {
            for (int j = 0; j < numHiddenLayer1; j++) {
                float updatedValue = W_h1h2[j][i] + TRAINING_RATE * hidden2_errors[i] * hidden1[j];
                W_h1h2[j][i] = clampWeight(updatedValue);
            }
        }

// update the input to hidden1 weights:
        for (int i = 0; i < numHiddenLayer1; i++) {
            for (int j = 0; j < numInputLayer; j++) {
                float updatedValue = W_ih1[j][i] + TRAINING_RATE * hidden1_errors[i] * inputs[j];
                W_ih1[j][i] = clampWeight(updatedValue);
            }
        }

        for (int i = 0; i < numOutputLayer; i++) {
            error += Math.abs(outs[i] - outputs[i]);
        }

        currentExample++;
        if (currentExample >= num_cases) {
            currentExample = 0;
//            currentExample=currentExample%num_cases;
        }
        return error;
    }

    private float clampWeight(float weight) {
        float ret;
        if (weight < -10) {
            ret = -10;
        } else if (weight > 10) {
            ret = 10;
        } else {
            ret = weight;
        }
        return ret;
    }

    private float sigmoidDerived(float x) {
        float funcRez = sigmoid(x);
        return funcRez * (1.0f - funcRez);
    }

    private float sigmoid(float x) {
        return (float) (1.0f / (1.0f + Math.exp(-x)));
    }

    private void randomizeWeight(float for1, float for2, float W[][]) {
        for (int i = 0; i < for1; i++) {
            for (int j = 0; j < for2; j++) {
                W[i][j] = (float) (2f * Math.random() - 1.f);
            }
        }
    }

    private void printResults(float[] input, float[] output) {
        float[] trainingResults = recall(input);
        System.out.print("Test case: ");
        for (int i = 0; i < input.length; i++) {
            System.out.print(printFormat(input[i]));
        }
        System.out.print("   ");
        System.out.print("Defined output case: ");
        for (int i = 0; i < output.length; i++) {
            System.out.print(printFormat(output[i]));
        }
        System.out.print("   ");
        System.out.print("Training results: ");
        for (int i = 0; i < output.length; i++) {
            System.out.print(printFormat(trainingResults[i]));
        }
        System.out.println();
    }

    private String printFormat(float x) {
        String ret = "" + x + "00";
        int numOfDecimals = ret.indexOf(".");
        if (numOfDecimals > -1) {
            ret = ret.substring(0, numOfDecimals + 3);
        }
//      if not starts with minus center
        if (!ret.startsWith("-")) {
            ret = " " + ret;
        }
        return ret;
    }

}
