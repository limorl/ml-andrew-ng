function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testLinearRegCostFunction();
  testPolyFeatures();
  
  #################################################################
  # Tests
  #################################################################
  function testLinearRegCostFunction()
    fprintf("testing linearRegCostFunction...");
    numTests = 0;
    numSuccess = 0;
    
    # test case
    X = [[1 1 1]' magic(3)];
    y = [7 6 5]';
    theta = [0.1; 0.2; 0.3; 0.4];
    lambda = 0;
    
    expected_J = 1.3533;
    expected_g = [-1.4000; -8.7333; -4.3333; -7.9333];
    
    [J g] = linearRegCostFunction(X, y, theta, lambda);
    numTests = numTests + 1;
    s1 = validateNum(expected_J, J, "testLinearRegCostFunction", numTests);
    s2 = validateVector(expected_g, g, "testLinearRegCostFunction", numTests);
    numSuccess = numSuccess + 1;
    
    # test case
    X = [[1 1 1]' magic(3)];
    y = [7 6 5]';
    theta = [0.1; 0.2; 0.3; 0.4];
    lambda = 7;
    
    expected_J = 1.6917;
    expected_g = [-1.400; -8.2667; -3.6333; -7.000];
    
    [J g] = linearRegCostFunction(X, y, theta, lambda);
    numTests = numTests + 1;
    s1 = validateNum(expected_J, J, "testLinearRegCostFunction", numTests);
    s2 = validateVector(expected_g, g, "testLinearRegCostFunction", numTests);
    numSuccess = numSuccess + 1;
   
   #test case
   X = [1 2 3 4];
   y = 5;
   theta = [0.1 0.2 0.3 0.4]';
   lambda = 7;
   
   expected_J = 3.0150;
   expected_g = [-2.0000; -2.6000; -3.9000; -5.2000];
    
   [J g] = linearRegCostFunction(X, y, theta, lambda);
   numTests = numTests + 1;
   s1 = validateNum(expected_J, J, "testLinearRegCostFunction", numTests);
   s2 = validateVector(expected_g, g, "testLinearRegCostFunction", numTests);
   numSuccess = numSuccess + 1;
  
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testPolyFeatures()
    fprintf("testing testPolyFeatures...");
    numTests = 0;
    numSuccess = 0;
    
     # test case
    X = [1; 1; 1];
    p = 1;
    
    expectedX_poly = [1; 1; 1];
    
    X_poly = polyFeatures(X,p);
    numTests = numTests + 1;
    numSuccess =  numSuccess + validateMatrix(expectedX_poly, X_poly, "testPolyFeatures", numTests);
    
    # test case
    X = [1; 1; 1];
    p = 3;
    
    expectedX_poly = [1 1 1; 1 1 1; 1 1 1];
    
    X_poly = polyFeatures(X,p);
    numTests = numTests + 1;
    numSuccess =  numSuccess + validateMatrix(expectedX_poly, X_poly, "testPolyFeatures", numTests);
    
     # test case
    X = [1; 2; 3];
    p = 3;
    
    expectedX_poly = [1 1 1; 2 4 8; 3 9 27];
    
    X_poly = polyFeatures(X,p);
    numTests = numTests + 1;
    numSuccess =  numSuccess + validateMatrix(expectedX_poly, X_poly, "testPolyFeatures", numTests);
    
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

 
 
