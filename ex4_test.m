function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testDigitsToVec();
  testSigmoidGardient();
  testNnCostFunction();
  
  #################################################################
  # Tests
  #################################################################
  function testDigitsToVec()
    fprintf("testing digitsToVec...");
    numTests = 0;
    numSuccess = 0;
    
    # test case
    y = [1; 2; 3; 4 ;5 ;6 ;7 ;8; 9; 0];
    expected_Y = ...
    [1 0 0 0 0 0 0 0 0 0;...
     0 1 0 0 0 0 0 0 0 0;...
     0 0 1 0 0 0 0 0 0 0;...
     0 0 0 1 0 0 0 0 0 0;...
     0 0 0 0 1 0 0 0 0 0;...
     0 0 0 0 0 1 0 0 0 0;...
     0 0 0 0 0 0 1 0 0 0;...
     0 0 0 0 0 0 0 1 0 0;...
     0 0 0 0 0 0 0 0 1 0;...
     0 0 0 0 0 0 0 0 0 1];
     
    Y = digitsToVec(y,10);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected_Y, Y, "testDigitsToVec", numTests);
     
    fprintf("done (%d/%d).\n", numSuccess, numTests);
    
    # test case
    y = [4; 2; 3];
    expected_Y = ...
    [0 0 0 1;...
     0 1 0 0;...
     0 0 1 0];
     
    Y = digitsToVec(y,4);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected_Y, Y, "testDigitsToVec", numTests);
     
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testNnCostFunction()
    fprintf("testing nnCostFunction...");
    numTests = 0;
    numSuccess = 0;
    
    # nueral network
    il = 2;              % input layer
    hl = 2;              % hidden layer
    nl = 4;              % number of labels
    nn = [ 1:18 ] / 10;  % nn_params
    
    # test case without regularization
    X = cos([1 2 ; 3 4 ; 5 6]);
    y = [4; 2; 3];
    expected_J =  7.4070;
    expected_grad = [0.766138; 0.979897; -0.027540; -0.035844; -0.024929; -0.053862; 0.883417; 0.568762; 0.584668; 0.598139; 0.459314; 0.344618; 0.256313; 0.311885; 0.478337; 0.368920; 0.259771; 0.322331] ;
 
    [J grad] = nnCostFunction(nn, il, hl, nl, X, y, 0);
    numTests = numTests + 1;
    success1 = validateNum(expected_J, J, "nnCostFunction", numTests);
    success2 = validateVector(expected_grad, grad, "nnCostFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);

   # test case with regularization
    lambda = 4;
    expected_J = 19.474;
    expected_grad = [0.76614; 0.97990; 0.37246; 0.49749; 0.64174; 0.74614; 0.88342; 0.56876; 0.58467; 0.59814; 1.92598; 1.94462; 1.98965; 2.17855; 2.47834; 2.50225; 2.52644; 2.72233];
    
    [J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)
    numTests = numTests + 1;
    success1 = validateNum(expected_J, J, "nnCostFunction", numTests);
    success2 = validateVector(expected_grad, grad, "nnCostFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  
  function testSigmoidGardient()
    fprintf("testing testSigmoidGardient...");
    numTests = 0;
    numSuccess = 0;
    
    # test case
    X = [[-1 -2 -3] ; magic(3)];
    expected_g = [1.9661e-001  1.0499e-001  4.5177e-002; 3.3524e-004  1.9661e-001  2.4665e-003; 4.5177e-002  6.6481e-003  9.1022e-004; 1.7663e-002  1.2338e-004  1.0499e-001];
    
    g = sigmoidGradient(X);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected_g, g, "testSigmoidGardient", numTests);
    
    # test case 
    X = [0.054017   0.166433;...
        -0.523820  -0.588183;...
         0.665184   0.889567];
    expected_g = [0.24982   0.24828;...
                  0.23361   0.22957;...
                  0.22426   0.20640];

    g = sigmoidGradient(X);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateMatrix(expected_g, g, "testSigmoidGardient", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  #{
  function testNnCostFunction()
    fprintf("testing nnCostFunction...");
    numTests = 0;
    numSuccess = 0;
    
    # nueral network
    il = 2;              % input layer
    hl = 2;              % hidden layer
    nl = 4;              % number of labels
    nn = [ 1:18 ] / 10;  % nn_params
    
    # test case without regularization
    X = cos([1 2 ; 3 4 ; 5 6]);
    y = [4; 2; 3];
    expected_J =  7.4070;
    expected_grad = [0.766138; 0.979897; -0.027540; -0.035844; -0.024929; -0.053862; 0.883417; 0.568762; 0.584668; 0.598139; 0.459314; 0.344618; 0.256313; 0.311885; 0.478337; 0.368920; 0.259771; 0.322331] ;
 
    [J grad] = nnCostFunction(nn, il, hl, nl, X, y, 0);
    numTests = numTests + 1;
    success1 = validateNum(expected_J, J, "nnCostFunction", numTests);
    success2 = validateVector(expected_grad, grad, "nnCostFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);

   # test case with regularization
    lambda = 4;
    expected_J = 19.474;
    expected_grad = [0.76614; 0.97990; 0.37246; 0.49749; 0.64174; 0.74614; 0.88342; 0.56876; 0.58467; 0.59814; 1.92598; 1.94462; 1.98965; 2.17855; 2.47834; 2.50225; 2.52644; 2.72233];
    
    [J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)
    numTests = numTests + 1;
    success1 = validateNum(expected_J, J, "nnCostFunction", numTests);
    success2 = validateVector(expected_grad, grad, "nnCostFunction", numTests);
    numSuccess = numSuccess + (success1 && success2);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  #}
  
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

 
 
