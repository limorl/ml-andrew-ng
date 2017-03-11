function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testSigmoid();
  testCostFunction();
  testPredict();
  testCostFunctionReg();
  
  #################################################################
  # Tests
  #################################################################
  function testSigmoid()
    fprintf("testing sigmoid...");
    numTests = 0;
    numSuccess = 0;
    
    val = 1200000;
    expected = 1;
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "sigmoid", numTests);
    
    val = -25000;
    expected = 0;
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "sigmoid", numTests);
    
    val = 0;
    expected = 0.5;
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "sigmoid", numTests);
    
    val = [4 5 6];
    expected = [0.9820 0.9933 0.9975];
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected, actual, "sigmoid", numTests);
    
    val = magic(3);
    expected = [0.9997 0.7311 0.9975; 0.9526 0.9933 0.9991; 0.9820 0.9999 0.8808 ];
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected, actual, "sigmoid", numTests);
    
    val = eye(2);
    expected = [0.7311 0.5000; 0.5000 0.7311];
    actual = sigmoid(val);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected, actual, "sigmoid", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
   function testCostFunction()
    fprintf("testing costFunction...");
    
    numTests = 0;
    numSuccess = 0;
    
    X = [ones(3,1) magic(3)];
    y = [1; 0; 1;];
    theta = [-2; -1; 1; 2];
    
    % un-regularized (regularization factor = 0)
    [actual_j actual_g] = costFunction(theta, X, y);
    expected_j = 4.6832;
    expected_g = [0.31722; 0.87232; 1.64812; 2.23787];
    numTests = numTests + 1;
    success1 = validateNum(expected_j, actual_j, "cosFunction", numTests);
    success2 = validateVector(expected_g, actual_g, "cosFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);

    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testPredict()
    fprintf("testing predict...");
    numTests = 0;
    numSuccess = 0;
    
    X = [1 1 ; 1 2.5 ; 1 3 ; 1 4];
    theta = [-3.5 ; 1.3];
    expected = [0; 0; 1; 1];
    actual = predict(theta, X);
    
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected, actual, "predict", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testCostFunctionReg()
    fprintf("testing costFunctionReg...");
    
    numTests = 0;
    numSuccess = 0;
    
    X = [ones(3,1) magic(3)];
    y = [1; 0; 1];
    theta = [-2; -1; 1; 2];
      
    % un-regularized (regularization factor = 0)
    [actual_j actual_g] = costFunction(theta, X, y, 0);
    expected_j = 4.6832;
    expected_g = [0.31722; 0.87232; 1.64812; 2.23787];
    numTests = numTests + 1;
    success1 = validateNum(expected_j, actual_j, "cosFunctionReg", numTests);
    success2 = validateVector(expected_g, actual_g, "cosFunctionReg", numTests);
    numSuccess = numSuccess + (success1 && success2);
    
     % un-regularized (regularization factor = 4)
    [actual_j actual_g] = costFunctionReg(theta, X, y, 4);
    expected_j = 8.6832;
    expected_g = [0.31722; -0.46102; 2.98146; 4.90454];
    numTests = numTests + 1;
    success1 = validateNum(expected_j, actual_j, "cosFunctionReg", numTests);
    success2 = validateVector(expected_g, actual_g, "cosFunctionReg", numTests);
    numSuccess = numSuccess + (success1 && success2);
    
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

