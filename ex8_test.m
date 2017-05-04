function test
  format long;
  epsilon = 0.00001;   # accuracy threshold
  
  testEstimateGaussian();
  testSelectThreshold();
  
  #################################################################
  # Tests
  #################################################################
  function testEstimateGaussian()
    testName = "estimateGaussian";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case - X is mx1 matrix
    x = [1; 2; 3];
    expected_mu = [2];
    expected_sigma2 = [2/3]; # sigma2 = (1/m)*((1-2)^2 + (2-2)^2 + (3-2)^2);
    
    [mu sigma2] = estimateGaussian(x);
    numTests += 1;
    
    s1 = validateNum(expected_mu, mu, testName, numTests);
    s2 = validateNum(expected_sigma2, sigma2, testName, numTests);
    numSuccess += s1 && s2;
  
    # test case 
    X = [1 2 3; 1 2 3; 1 2 3; 1 2 3; 1 2 3; 1 2 3];
    expected_mu = [1; 2; 3];
    expected_sigma2 = [0; 0; 0];  
    
    [mu sigma2] = estimateGaussian(X);
    numTests += 1;
    
    s1 = validateVector(expected_mu, mu, testName, numTests);
    s2 = validateVector(expected_sigma2, sigma2, testName, numTests);
    numSuccess += s1 && s2;
  
    # test case 
     X = [1 2 3 4; 4 3 2 1; 1 2 3 4; 4 3 2 1; 1 2 3 4; 4 3 2 1];
    expected_mu = [2.5; 2.5; 2.5; 2.5];
    expected_sigma2 = [2.25; 0.25; 0.25; 2.25];     
    
    [mu sigma2] = estimateGaussian(X);
    numTests += 1;
    
    s1 = validateVector(expected_mu, mu, testName, numTests);
    s2 = validateVector(expected_sigma2, sigma2, testName, numTests);
    numSuccess += s1 && s2;
   
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testSelectThreshold()
    testName = "testSelectThreshold";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case 
    yval = [0; 1; 1; 0];
    pval = [0.9; 0.09; 0.09; 0.9];  
    [bestEpsilon bestF1] = selectThreshold(yval, pval);
    numTests += 1;
    
    expected_bestEpsilon = 0.09081;
    expected_bestF1 = 1;
    
    s1 = validateNum(expected_bestEpsilon, bestEpsilon, testName, numTests);
    s2 = validateNum(expected_bestF1, bestF1, testName, numTests);
    numSuccess += s1 && s2;
      
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
