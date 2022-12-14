# graphPINN
A message-passing based PINN for emulating magnetohydrostatic models built on a graph-convolutional framework. See the [AGU poster for more information](http://agu2022fallmeeting-agu.ipostersessions.com/Default.aspx?s=DD-14-A6-2B-20-A2-07-93-67-66-15-A2-2A-F9-50-38)

This work is ongoing, but a link will be posted to the arXiv paper for this Physics-Informed Neural Net as soon as it exists; watch this space! In the mean time, check out the [scattered-note Magnetohydrostatic model this PINN is being trained to emulate](https://doi.org/10.1016/j.jcp.2022.111214)

This model was heavily inspired by the methodology in [Li et al 2020](https://doi.org/10.48550/arXiv.2003.03485)

---

The basic NN architecture defined in the Kernel and ConvGraph classes should be applicable to a wide range of neural net use-cases. ConvGraph also keeps track of the physical derivatives with respect to space, which is useful for PINN loss functions such as the Magnetohydrostatic loss function provided. The data submodule is built to read the training dataset available via S3 bucket on Heliocloud, but any pyg.data.Dataset should work, as long as it provides physical location, magnetic field and plasma forcing for each graph node.

This work is associated with the [NASA Center for HelioAnalytics](https://helioanalytics.io/)
