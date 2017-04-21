function test
  format long;
  epsilon = 0.0001;   # accuracy threshold
  
  testFindClosestCentroids();
  testComputeCentroids();
  
  #################################################################
  # Tests
  #################################################################
  function testFindClosestCentroids()
    testName = "findClosestCentroids";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case - single cluster
    X = [1 1; 2 2; 3 3 ; 0.5 0.5; 2.5 2.5; 3.5 3.5];
    centroids = [5 5];
    expected_idx = [1; 1; 1; 1; 1; 1;];
    
    idx = findClosestCentroids(X, centroids);
    numTests += 1;
    numSuccess += validateVector(expected_idx, idx, testName, numTests);
    
    # test case 
    X = [1 1; 2 2; 3 3 ; 0.5 0.5; 2.5 2.5; 3.5 3.5];
    centroids = [0.9 0.9; 1.9 1.9; 2.9  2.9];
    expected_idx = [1; 2; 3; 1; 2; 3;];
    
    idx = findClosestCentroids(X, centroids);
    numTests += 1;
    numSuccess += validateVector(expected_idx, idx, testName, numTests);
    
    # test case 
    X = [1 1; 2 2; 3 3 ; 4 4; 5 5; -1 -1; -2 -2; -3 -3 ; -4 -4; -5 -5];
    centroids = [1 1; -1 -1];
    expected_idx = [1; 1; 1; 1; 1; -1; -1; -1; -1; -1;];
    
    idx = findClosestCentroids(X, centroids);
    numTests += 1;
    numSuccess += validateVector(expected_idx, idx, testName, numTests);
    
    # test case
    X = [1.84207953112616   4.60757160448228;...
         5.65858312061882   4.79996405444154;...
         6.35257892020234   3.29085449875427];
    centroids = [3 3; 6 2; 8 5];
    expected_idx = [1; 3; 2];
    
    idx = findClosestCentroids(X, centroids);
    numTests += 1;
    numSuccess += validateVector(expected_idx, idx, testName, numTests);
  
   
   fprintf("done (%d/%d).\n", numSuccess, numTests);
  end
  
  function testComputeCentroids()
    testName = "testComputeCentroids";
    
    fprintf("testing %s...", testName);
    numTests = 0;
    numSuccess = 0;
    
    # test case - single cluster
    X = [1 1; 2 2; 3 3; 4 4; 5 5];
    idx = [1; 1; 1; 1; 1];
    K = 1;
    
    expected_centroids = [3 3];
    centroids = computeCentroids(X, idx, K);
    
    numTests += 1;
    numSuccess += validateVector(expected_centroids, centroids, testName, numTests);
   
    # test case - two clusters
    X = [1 1 1; 2 2 2; 3 3 3; 5 5 5; 6 6 6; 7 7 7];
    idx = [1; 1; 1; 2; 2; 2];
    K = 2;
    
    expected_centroids = [2 2 2; 6 6 6];
    centroids = computeCentroids(X, idx, K);
    
    numTests += 1;
    numSuccess += validateVector(expected_centroids, centroids, testName, numTests);
      
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

 
 
