use super::Tensor;
use ndarray_layout::ArrayLayout;
impl<T, const N: usize> Tensor<T, N> {
    pub fn index(self, indices: &[usize]) -> Self {
        let mut layout = self.layout.clone();
        for &index in indices {
            layout = layout.index(0, index)
        }
        Self {
            dt: self.dt,
            layout,
            data: self.data,
        }
    }

    pub fn merge(self, axis: usize, len: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.merge_be(axis, len).unwrap(),
            data: self.data,
        }
    }

    pub fn tile(self, axis: usize, tiles: &[usize]) -> Self {
        self.map_layout(|l| l.tile_be(axis, tiles))
    }

    pub fn broadcast(self, axis: usize, times: usize) -> Self {
        self.map_layout(|l| l.broadcast(axis, times))
    }

    pub fn transpose(self, perm: &[usize]) -> Self {
        self.map_layout(|l| l.transpose(perm))
    }

    pub fn slice(self, axis: usize, start: usize, len: usize) -> Self {
        self.map_layout(|l| l.slice(axis, start, 1, len))
    }

    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a
    where
        T: Clone,
    {
        let self_clone = self.clone();
        self.layout
            .split(axis, parts)
            .map(move |layout| self_clone.clone().map_layout(|_| layout))
    }

    fn map_layout(mut self, f: impl FnOnce(&ArrayLayout<N>) -> ArrayLayout<N>) -> Self {
        self.layout = f(&self.layout);
        self
    }
}
