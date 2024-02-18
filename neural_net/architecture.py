class Sequential:
    def __init__(self,steps):
        self.steps = steps
    def eval(self):
        self.out  = self.steps[0].eval()
        for step in self.steps[1:]:
            step.In = self.out
            step.eval()
            self.out = step.out
        return self.out
    def train(self,X,y,α,n_epochs,cost,metrics,batch_size=None):

        n = len(y)
        batch_size = batch_size or n
        batchix = list(range(0,n,batch_size))
        if batchix[-1]<n : batchix.append(n)
        batchix = [slice(low,high) for low,high in zip(batchix,batchix[1:])]

        for n in range(n_epochs):
            for ix in batchix:
                new_y,new_X = y[ix,:],X[ix,:]
                self.steps[0].In = [(id(new_X),new_X)]
                outid,p = self.eval()[0]
                L,m = cost(new_y,p),metrics(new_y,p)
                Δ =  {outid:α*L.prime()}
                self.update(Δ)
                print('epoch',n,'batch',ix,'metrics',m.eval(),'cost',L.eval())
    def update(self,Δnext):
        self.Δnext = Δnext
        for step in self.steps[::-1]:
            self.Δnext = step.update(self.Δnext)
        return self