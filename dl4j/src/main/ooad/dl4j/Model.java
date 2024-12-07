package ooad.dl4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface Model {
    void fit(DataSetIterator data);
    INDArray predict(INDArray input);
}