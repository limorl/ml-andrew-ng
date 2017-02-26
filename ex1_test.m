function test
  format long;
  epsilon = 0.00000000001;
  
  testComputeCost(); 
  testGradientDescent();
  testComputeCostMulti();
  testFeatureNormalize();
  testGradientDescentMulti();
  testNormalEqn();
  
  #################################################################
  # Tests
  #################################################################
  function testComputeCost()
    fprintf("testing computeCost()...");
    
    A = [1 10; 1 20; 1 30];
    b = [35; 65; 95];
    m = length(b);
    numTests = 0;
    numSuccess = 0;

    theta = [5; 3];
    expected = 0;
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCost", numTests);

    theta = [0; 0];
    expected = ((-35)^2 + (-65)^2 + (-95)^2)/(2*m);
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCost", numTests);
    
    theta = [0; 4];
    expected = (5^2 + 15^2 + 25^2)/(2*m);
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCost", numTests);
    
    theta = [100; 0];
    expected = (65^2 + 35^2 + 5^2)/(2*m);
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCost", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testComputeCostMulti()
    fprintf("testing computeCostMulti()...");

    A = [1 10 5; 1 20 10; 1 30 15; 1 40 20; 1 50 25];
    b = [20; 35; 50; 65; 80];
    m = length(b);
    
    numTests = 0;
    numSuccess = 0;

    theta = [5; 2; -1];
    expected = 0;
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCostMulti", numTests);

    theta = [0; 0; 0];
    expected = ((-20)^2 + (-35)^2 + (-50)^2 + (-65)^2 + (-80)^2)/(2*m);
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCostMulti", numTests);
    
    theta = [5; 1; 1;];
    expected = 0;
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCostMulti", numTests);
    
    theta = [8; 1; 1;];
    expected = (3^2 + 3^2 + 3^2 + 3^2 + 3^2)/(2*m);
    actual = computeCost(A,b, theta);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateNum(expected, actual, "computeCostMulti", numTests);

    fprintf("done (%d/%d)\n", numSuccess, numTests);
  end
  
  function testGradientDescent() 
    fprintf("testing gradientDescent()...");

    A = [1 10; 1 20; 1 30];
    b = [35; 65; 95];
    m = length(b);
    alpha = 0.01;
    
    %delta = (1/m)*X'*(X*theta-y);
    %theta = theta - alpha*delta;
    
    numTests = 0;
    numSuccess = 0; 
    
    startTheta = [5; 3]; % This is the optimum theta
    expectedTheta = [5; 3];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    
    startTheta = [5; 3]; % This is the optimum theta
    expectedTheta = [5; 3];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 5);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);

    startTheta = [5; 3];
    expectedTheta = [5; 3];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1500);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    
    theta = [0; 0];
    expectedTheta = [(0-alpha*(1/m)*(-35 -65 -95)); (0-alpha*(1/m)*(-35*10 -65*20 -95*30))];
    [actualTheta, actualJ_history] = gradientDescent(A, b, theta, alpha, 1);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    startTheta = [0; 0];
    expectedTheta = [5; 3];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1500);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    E = [1 10; 1 20; 1 30; 1 40; 1 50; 1 60; 1 70; 1 80; 1 90; 1 100];
    f = [80; 60; 40; 20; 0; -20; -40; -60; -80; -100];
    m = length(f);
    
    startTheta = [0; 0];
    expectedTheta = [(0-alpha*(1/m)*(-80 -60 -40 -20 +0 +20 +40 +60 +80 +100)); (0-alpha*(1/m)*(-80*10 -60*20 -40*30 -20*40 +0*50 +20*60 +40*70 +60*80 +80*90 +100*100))];
    [actualTheta, actualJ_history] = gradientDescent(E, f, startTheta, alpha, 1);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    startTheta = [5; 5];
    expectedTheta = [100; -2];
    [actualTheta, actualJ_history] = gradientDescent(E, f, startTheta, alpha, 1500);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescent", numTests);
    
    fprintf("done (%d/%d)\n", numSuccess, numTests);
  end
  
  function testFeatureNormalize()
    fprintf("testing featureNomalize()...");
    
   A = [1 2 3 ; 1 4 6; 1 6 9];
   
   numTests = 0;
   numSuccess = 0;
   
   mu = [1; 4; 6];
   sigma = [0; 2; 3];
   ANorm = [0 -2/2  -3/3; 0 0 0; 0 2/2 3/3];
   
   [actANorm, actMu, actSigma] = featureNormalize(A);
   numTests = numTests + 1;
   numSuccess = numSuccess + validateMatrix(ANorm, actANorm, "featureNormalize", numTests);
   numSuccess = numSuccess + validateVector(mu, actMu, "featureNormalize", numTests);
   numSuccess = numSuccess + validateVector(sigma, actSigma, "featureNormalize", numTests);   
   
   fprintf("done (%d/%d).\n", numSuccess > 0, numTests);
  end
  
  function testGradientDescentMulti()
    fprintf("testing gradientDescentMulti()...");
     
    A = [1 10 5; 1 20 10; 1 30 15; 1 40 20; 1 50 25];
    b = [20; 35; 50; 65; 80];
    m = length(b);
    alpha = 0.01;
    
    # test without normalizing A
    numTests = 0;
    numSuccess = 0; 
    
    startTheta = [5; 2; -1]; % This is the optimum theta
    expectedTheta = [5; 2; -1];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescentMulti", numTests);
    
    startTheta = [5; 2; -1]; % This is the optimum theta
    expectedTheta = [5; 2; -1];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 5);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescentMulti", numTests);

    startTheta = [5; 2; -1];
    expectedTheta = [5; 2; -1];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1500);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescentMulti", numTests);
    
    theta = [0; 0; 0];
    expectedTheta = [(0-alpha*(1/m)*(-20 -35 -50 -65 -80)); (0-alpha*(1/m)*(-20*10 -35*20 -50*30 -65*40 -80*50)); (0-alpha*(1/m)*(-20*5 -35*10 -50*15 -65*20 -80*25))];
    [actualTheta, actualJ_history] = gradientDescent(A, b, theta, alpha, 1);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescentMulti", numTests);
    
    startTheta = [0; 0; 0];
    expectedTheta = [5; 2; -1];
    [actualTheta, actualJ_history] = gradientDescent(A, b, startTheta, alpha, 1500);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "gradientDescentMulti", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testNormalEqn()
    fprintf("testing normalEqn()...");
     
    A = [1 10 5; 1 20 10; 1 30 15];
    b = [20; 35; 50];
    m = length(b);
    
    # test without normalizing A
    numTests = 0;
    numSuccess = 0; 
    
    expectedTheta = [5; 1.5; 0];
    [actualTheta] = normalEqn(A,b);
    numTests = numTests + 1;
    numSuccess = numSuccess + validateVector(expectedTheta, actualTheta, "normalEqn", numTests);
    
    fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  ##################################################################
  # Validation functions
  ##################################################################
  function success = validateNum(expected, actual, msg, index)
    success = 1;
    if(abs(expected - actual) > epsilon)
        fprintf("%s(%d) FAIL: %d instead of %d \n", msg, index, actual, expected);
        success = 0;
    end
  end
  
  function success = validateVector(expected, actual, msg, index)
    success = 1;
    if(length(expected) != length(actual))
      fprintf("%s(%d) FAIL: vectors not of the same size. expected: %d, actual: %d \n", msg, index, length(expected), length(actual));
      success = 0;
    elseif(sum(expected-actual) > epsilon)
        fprintf("%s(%d) FAIL: vector not as expected.\n expected: %s, actual: %s\n", msg, index, mat2str(expected), mat2str(actual)); 
        success = 0;
    end
  end
 
  function success = validateMatrix(expected, actual, msg, index)
    success = 1;
    if(size(expected,1) != size(actual,1) || size(expected,2) != size(actual,2))
      fprintf("%s(%d) FAIL: matrices are not of the same size. expected: %dx%d, actual: %dx%d \n", msg, index, size(expected,1), size(expected,2), size(actual,1), size(actual,2));
      success = 0;
    else
      D = expected - actual;
      for i = 1:size(expected,2)
        if(sum(D(:,i)) != 0)
          fprintf("%s(%d) FAIL: matrix not as expected.\n expected: %s, actual: %s\n", msg, index, mat2str(expected), mat2str(actual));
          success = 0;
        end
      end 
    end
  end  
end
