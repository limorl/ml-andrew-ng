function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testLrCostFunction();
  testOneVsAll();
  testPredictOneVsAll();
  
  #################################################################
  # Tests
  #################################################################
  function testLrCostFunction()
    fprintf("testing lrCostFunction...");
    numTests = 0;
    numSuccess = 0;
    
    theta = [-2; -1; 1; 2];
    X = [ones(5,1) reshape(1:15,5,3)/10];
    y = [1;0;1;0;1] >= 0.5;       % creates a logical array
    lambda = 3;
    expected_J = 2.5348;
    expected_grad = [0.14656; -0.54856; 0.72472; 1.39800];
    
    [J grad] = lrCostFunction(theta, X, y, lambda);
    numTests = numTests + 1;
    
    success1 = validateNum(expected_J, J, "lrCosFunction", numTests);
    success2 = validateVector(expected_grad, grad, "lrCosFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testOneVsAll()
    fprintf("testing oneVsAll...");
    numTests = 0;
    numSuccess = 0;
    
    X = [magic(3) ; sin(1:3); cos(1:3)];
    y = [1; 2; 2; 1; 3];
    num_labels = 3;
    lambda = 0.1;
    expected_all_theta = [-0.559478   0.619220  -0.550361  -0.093502; -5.472920  -0.471565   1.261046   0.634767; 0.068368  -0.375582  -1.652262  -1.410138];
    
    [all_theta] = oneVsAll(X, y, num_labels, lambda)
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected_all_theta, all_theta, "oneVsAll", numTests);

    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testPredictOneVsAll()
    fprintf("testing predictOneVsAll...");
    numTests = 0;
    numSuccess = 0;
    
    # test case
    all_theta = [1 -6 3; -2 4 -3];
    X = [1 7; 4 5; 7 8; 1 4];
    expected_p = [1; 2; 2; 1];
    
    actual_p = predictOneVsAll(all_theta, X)
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_p, actual_p, "predictOneVsAll", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testPredict()
    fprintf("testing predict...");
    numTests = 0;
    numSuccess = 0;
    
    # test case
    Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
    Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
    X = reshape(sin(1:16), 8, 2);
    
    expected_p = [4; 1; 1; 4; 4; 4; 4; 2];
    
    p = predict(Theta1, Theta2, X);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expected_p, p, "predict", numTests); 

   #{
   a3 =
   0.53036   0.54588   0.55725   0.56352
   0.54459   0.54298   0.53754   0.52875
   0.49979   0.49616   0.49288   0.49024
   0.41357   0.42199   0.43736   0.45844
   0.37321   0.40368   0.44349   0.48911
   0.42073   0.45935   0.50210   0.54464
   0.50962   0.53216   0.55173   0.56659
   0.54882   0.55033   0.54738   0.54021
   #}
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

 
 
