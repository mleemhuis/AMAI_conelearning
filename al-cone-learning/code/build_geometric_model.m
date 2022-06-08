function[geom_model] = build_geometric_model(attr_space)
    % number of concepts
    n_conc = size(attr_space, 1);

    % different possibilities of creating the geometric model: here: Hamming Distance
    ham_dis = pdist2(attr_space, attr_space, 'hamming');

    % get the pairs with the highest hamming distances
    % used as python function
    matching = py.maxweightmod.maxweight(py.numpy.array(ham_dis));

    % assign one half-axis to each atomic concept
    % initialize geometric model
    geom_model = zeros(n_conc, ceil(n_conc / 2));

    for i=1:matching.length
        x = cell(matching{i});
        geom_model(double(x{1})+1, i) = 1;
        geom_model(double(x{2})+1, i) = -1;
    end