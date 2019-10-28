
            lc = DefineNumber[ 0.000975, Name "Parameters/lc" ];
            H = 100.0e-3;
            L = 100.0e-3;

            a = 25.0e-3;

            Point(1) = {0, 0, 0, lc};
            Point(2) = {L, 0, 0, lc};
            
            Point(3) = {L, a, 0, lc};
            Point(4) = {L, H, 0, lc};
            
            Point(5) = {0, H, 0, lc};
            Point(6) = {0, a, 0, lc};

            
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 6};
            Line(4) = {6, 1};
            Line Loop(5) = {1, 2, 3, 4};
            
            Line(6) = {3, 4};
            Line(7) = {4, 5};
            Line(8) = {5, 6};
            Line Loop(9) = {-3, 6, 7, 8};
            
            Plane Surface(10) = {5};
            Plane Surface(11) = {9};
            
            Physical Surface(1) = {10};
            Physical Surface(2) = {11};

