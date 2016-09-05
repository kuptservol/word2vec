package ru.skuptsov;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

/**
 * @author Sergey Kuptsov
 * @since 04/09/2016
 */
public class Word2VecRawTextTrainedByGoogleCorpus {

    public static void main(String[] args) throws Exception {

        File gModel = new ClassPathResource("GoogleNews-vectors-negative300.bin.gz").getFile();

        WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(gModel, true);

        System.out.println("Closest Words:");

        Collection<String> kingList = wordVectors.wordsNearest(Arrays.asList("king", "woman"), Arrays.asList("queen"), 10);
        System.out.println(kingList);
    }
}
