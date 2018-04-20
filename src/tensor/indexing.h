#ifndef INDEXING_H
#define INDEXING_H

#include <cassert>

struct range {
    int start, end;
    range(): start(-1), end(-1){}
    range(int _start, int _end): start(_start), end(_end) {}
    bool isset(){ return start != -1; }
    int size() const { assert(end !=-1); return end - start; }
};

struct block {
    range row, col;
    block(int x): col(-1, -1){
        (*this)(x);
    }

    block(int x, int y): col(-1, -1){
        (*this)(x, y);
    }

    block operator()(range r){
        assert (not (row.isset() and col.isset()));

        if (not row.isset()) { row = r; }
        else { col = r; }
        return *this;
    }

    block operator()(int x){
        return (*this)(x, x+1);
    }

    block operator()(int x, int y){
        assert (not (row.isset() and col.isset()));
        range r;
        r.start = x, r.end = y;
        return (*this)(r);
    }
};

#endif
