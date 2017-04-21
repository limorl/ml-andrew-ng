function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testGaussianKernel();
  testProcessEmail();
  testEmailFeatures();
  testMinElementIndex();
  
  #################################################################
  # Tests
  #################################################################
  function testGaussianKernel()
    testName = "gaussianKernel";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case
    x1 = [1];
    x2 = [1];
    sigma = 1;
    expectedSimilarity = 1;
    
    similarity = gaussianKernel(x1, x2, sigma);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expectedSimilarity, similarity, testName, numTests);
    
    # test case
    x1 = [2];
    x2 = [0];
    sigma = 1;
    expectedSimilarity = e^(-2);
    
    similarity = gaussianKernel(x1, x2, sigma);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expectedSimilarity, similarity, testName, numTests);
    
    # test case
    x1 = [1 1 1 1 1];
    x2 = [0 1 2 3 4];
    sigma = 1;
    expectedSimilarity = e^(-7.5);
    
    similarity = gaussianKernel(x1, x2, sigma);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expectedSimilarity, similarity, testName, numTests);
    
     # test case
    x1 = [1 2 1];
    x2 = [0 4 -1];
    sigma = 2;
    expectedSimilarity = e^(-9/8);
    
    similarity = gaussianKernel(x1, x2, sigma);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expectedSimilarity, similarity, testName, numTests);
    
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testProcessEmail()
    testName = "processEmail";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
     # test case - none of the words exist in vocabList
    text = 'Beautiful!';
    expected_indices = [];  

    word_indices = processEmail(text);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_indices, word_indices, testName, numTests);
    
    # test case - one of the words doesn't exist in vocabList
    text = 'I am wonderful';
    expected_indices = [64; 1867];  # wonderful --> wonder

    word_indices = processEmail(text);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_indices, word_indices, testName, numTests);
    
    # test case
    text = 'anyon know how much it cost to host a web portal well it depend on how mani';
    expected_indices = [86 916 794 1077 883 370 1699 790 1822 1831 883 431 1171 794 1002]';

    word_indices = processEmail(text);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_indices, word_indices, testName, numTests);
   
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testEmailFeatures()
    testName = "emailFeatures";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    #{
    vocab = {a, b, c, d, e, f, g, h, i, j, k, l}; # list of size 12
    vocab_size = size(vocab,1);
    
    # test case - none of the words are in vocab
    text = 'm n o p q r s t';
    word_indices = [];
    expected_features = zeros(vocab_size,1);  

    features = emailFeatures(word_indices);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_features, features, testName, numTests);
   
    # test case - all words are in vocab
    text = 'a b c d e f g h i j k l';
    word_indices = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12];
    expected_features = ones(vocab_size,1);  

    features = emailFeatures(word_indices);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_features, features, testName, numTests);
    #}
    vocabList = getVocabList();
    vocabSize = length(vocabList);

   # test case - none of the words are in vocab
   word_indices =  [];
   expected_features = zeros(vocabSize);
   
   features = emailFeatures(word_indices);
   numTests = numTests + 1;
   numSuccess = numSuccess + validateVector(expected_features, features, testName, numTests);
   
    # test case - all words are in vocab
   text = "The quick brown fox jumped over the lazy dog";
   word_indices =  [60; 100; 33; 44; 10; 53; 60; 58; 5];
   expected_features = zeros(vocabSize,1);
   expected_features(60) = 1;
   expected_features(100) = 1;
   expected_features(33) = 1;
   expected_features(44) = 1;
   expected_features(10) = 1;
   expected_features(53) = 1;
   expected_features(58) = 1;
   expected_features(5) = 1;
   
   features = emailFeatures(word_indices);
   numTests = numTests + 1;
   numSuccess = numSuccess + validateVector(expected_features, features, testName, numTests);
    
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testMinElementIndex()
    function [min_row min_col] = minElementIndex(M)
      rows = size(M,1);
      cols = size(M,2);
      
      [min_val min_index] = min(M(:)); # flat matrix is [col1; col2;...] 
      min_row = mod(min_index, rows);
      if min_row == 0 
        min_row = rows;
      end
      
      if mod(min_index, rows) == 0
        min_col = min_index/rows;
      else
        min_col = floor(min_index/rows) + 1;
      end
    end  
   
    testName = "minElementIndex";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case
    M = [1];
    expected_min_row = 1;
    expected_min_col = 1;
    
    [min_row min_col] = minElementIndex(M);
    numTests = numTests + 1;
    s1 = validateNum(expected_min_row, min_row, testName, numTests);
    s2 = validateNum(expected_min_col, min_col, testName, numTests);
    numSuccess = numSuccess + (s1 && s2);
    
    # test case
    M = [1 2 3; 4 5 6; 7 8 9];
    expected_min_row = 1;
    expected_min_col = 1;
    
    [min_row min_col] = minElementIndex(M);
    numTests = numTests + 1;
    s1 = validateNum(expected_min_row, min_row, testName, numTests);
    s2 = validateNum(expected_min_col, min_col, testName, numTests);
    numSuccess = numSuccess + (s1 && s2);
    
    # test case
    M = [1 2 3 3; 4 5 6 6; 7 8 9 0];
    expected_min_row = 3;
    expected_min_col = 4;
    
    [min_row min_col] = minElementIndex(M);
    numTests = numTests + 1;
    s1 = validateNum(expected_min_row, min_row, testName, numTests);
    s2 = validateNum(expected_min_col, min_col, testName, numTests);
    numSuccess = numSuccess + (s1 && s2);
    
    # test case
    M = [1 2 3 3; 4 5 0 6; 7 8 9 10];
    expected_min_row = 2;
    expected_min_col = 3;
    
    [min_row min_col] = minElementIndex(M);
    numTests = numTests + 1;
    s1 = validateNum(expected_min_row, min_row, testName, numTests);
    s2 = validateNum(expected_min_col, min_col, testName, numTests);
    numSuccess = numSuccess + (s1 && s2);
    
   fprintf("done (%d/%d).\n", numSuccess, numTests);

  end
  
  ##################################################################
  # Validation functions
  ##################################################################
  function success = validateNum(expected, actual, msg, index)
    success = 1;
    if(abs(expected - actual) > epsilon)
        fprintf("%s #%d FAIL: %d instead of %d \n", msg, index, actual, expected);
        success = 0;
    end
  end
  
  function success = validateString(expected, actual, msg, index)
    success = 1;
    if(strcmp(expected, actual) != 1)
        fprintf("%s #%d FAIL: %s instead of %s \n", msg, index, actual, expected);
        success = 0;
    end
  end
  
  function success = validateVector(expected, actual, msg, index)
    success = 1;
    if(length(expected) != length(actual))
      fprintf("%s #%d FAIL: vectors not of the same size. expected: %d, actual: %d \n", msg, index, length(expected), length(actual));
      success = 0;
    elseif(sum(expected-actual) > epsilon)
        fprintf("%s #%d FAIL: vector not as expected.\n expected: %s, actual: %s\n", msg, index, mat2str(expected), mat2str(actual)); 
        success = 0;
    end
  end
 
  function success = validateMatrix(expected, actual, msg, index)
    success = 1;
    if(size(expected,1) != size(actual,1) || size(expected,2) != size(actual,2))
      fprintf("%s #%d FAIL: matrices are not of the same size. expected: %dx%d, actual: %dx%d \n", msg, index, size(expected,1), size(expected,2), size(actual,1), size(actual,2));
      success = 0;
    else
      D = abs(expected - actual) >= epsilon;
      for i = 1:size(expected,2)
        if(sum(D(:,i)) != 0)
          fprintf("%s #%d FAIL: matrix not as expected.\n expected: %s, actual: %s\n", msg, index, mat2str(expected), mat2str(actual));
          success = 0;
        end
      end 
    end
  end  
end

 
 
