#include "linalg.h"
#include "utils/utils.h"

template <class _Tensor>
std::tuple<_Tensor, _Tensor, _Tensor> bidiagonalize(_Tensor A) {
    //std::cout << "begin bidiagonalize:\n";
    bool row_go, col_go;
    row_go = col_go = true;

    _Tensor Q_t(A.rows, A.rows);
    _Tensor B(A.rows, A.cols);
    _Tensor P(A.cols, A.cols);
    _Tensor x(0, 0);
    _Tensor y(0, 0);
    _Tensor v(0, 0);
    _Tensor H(0, 0);
    _Tensor K(0, 0);
    
    _Tensor subA(0, 0);
    _Tensor subQ_t(0, 0);
    _Tensor subP(0, 0);
    
    Q_t = identity(Q_t);
    P = identity(P);

    
    /* find length of diagonal and upper diagonal */
    int diag, udiag;
    if (A.rows < A.cols) {
        diag = A.rows;
        udiag = A.rows;
    } else {
        diag = A.cols;
        udiag = A.cols-1;
    }
    
    /* while loop runs till iterations for both rows and columns are exhausted.
       col loop is run length(diag) times and row loop is run length(udiag)
       times. in case of single element, hh multiplication is skipped         */
    int row_iter, col_iter;
    row_iter = col_iter = 0;
    while(row_go || col_go) {
        //std::cout << row_go << col_go << "\n";
        /* iterations annihilate below diagonal for cols */
        if(row_iter < diag) {
            /* slice x col out of A */
            block col = block(row_iter, A.rows)(row_iter, row_iter+1);
            x = slice(A, col);

            
            /* generate hh matrix based on x */
            if (x.rows > 1) {
                v = reflector(x);
                H = house(v);
                
                /* multiply H with relevant part of A from left */
                block rel = block(A.rows-H.rows, A.rows)(0, A.cols);
                subA = slice(A, rel);
                
                subA = H*subA;
                set_slice(A, rel, subA);
                
                // A = check_zeros(A);
                
                block rel2 = block(Q_t.rows-H.rows, Q_t.rows)(0, Q_t.cols);
                subQ_t = slice(Q_t, rel2);
                subQ_t = H*subQ_t;
                
                set_slice(Q_t, rel2, subQ_t);
            }
            
            row_iter++;
        } else {
            row_go = false;
        }
        
        /* iterations annihilate to the right of 1st upper diagonal for rows */
        if(col_iter < udiag) {      
            /* slice y row out of A */
            block row = block(col_iter, col_iter+1)(col_iter+1, A.cols);
            y = slice(A, row);

            if(y.cols > 1) {
                /* generate hh matrix based on y */
                v = reflector(y.transpose());
                K = house(v);
                
                /* multiply K with relevant part of A from right */
                block rel3 = block(0, A.rows)(A.cols-K.cols, A.cols);
                subA = slice(A, rel3);
                
                subA = subA*K;
                set_slice(A, rel3, subA);
                
                // A = check_zeros(A);
                
                block rel4 = block(0, P.rows)(P.cols-K.cols, P.cols);
                subP = slice(P, rel4);
                subP = subP*K;
                
                set_slice(P, rel4, subP);
            }
            
            col_iter++;
        } else {
            col_go = false;
        }
    }
    //B = check_zeros(A);
    B = A;
    
    auto products = std::make_tuple(Q_t, B, P);
    
    //std::cout << "end bidiagonalize:\n";
    return products;
}
