
            lc = DefineNumber[ 0.005, Name "Parameters/lc" ];
            H = 500.0e-3;
            L = 500.0e-3;
            t = 10.0e-3;
            
            

            Point(1) = {0, 0, 0, lc};
            Point(2) = {L, 0, 0, lc};
            
            Point(3) = {L, H, 0, lc};
            Point(4) = {0, H, 0, lc};
            

            
            Line(1) = {1, 2};
            Line(2) = {2, 3};
            Line(3) = {3, 4};
            Line(4) = {4, 1};
            Line Loop(1) = {1, 2, 3, 4};
            
            
            Plane Surface(1) = {1};
            
            out1[] = Extrude {0, 0, t}{Surface{1};};
            Compound Volume(2) = {1};
            
            Physical Surface(1) = {1};
            

