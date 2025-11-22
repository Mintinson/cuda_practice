#include <map>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <cutlass/cutlass.h>
// Cutlass command line parser
#include "cutlass/util/command_line.h"
#include "cutlass/coord.h"
#include "cutlass/util/reference/host/tensor_foreach.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"
/////////////////////////////////////////////////////

class Options
{
public:
    bool help;
    bool good;
    std::vector<int> extent;       ///< extent of tile to fill
    std::vector<int> stride;       ///< stride vector for layout function
    std::vector<int> output_shape; ///< output shape
    int vectorize;                 ///< sequences of consecutive output elements are concatenated into a vector
                                   ///  if, and only if, they were consecutive in source memory

public:
    /// Options
    Options() : help(false),
                good(true),
                extent({32, 8}),
                stride({32}),
                output_shape({16, 8}),
                vectorize(1)
    {
    }

    /// Constructs from command line parser
    Options(cutlass::CommandLine const &cmd_line) : help(false), good(true)
    {

        if (cmd_line.check_cmd_line_flag("help") ||
            cmd_line.check_cmd_line_flag("h"))
        {

            help = true;
        }

        if (cmd_line.check_cmd_line_flag("extent"))
        {
            cmd_line.get_cmd_line_arguments("extent", extent);
        }
        else
        {
            extent = {32, 8};
        }

        if (cmd_line.check_cmd_line_flag("stride"))
        {
            cmd_line.get_cmd_line_arguments("stride", stride);
        }

        int default_output_shape[] = {16, 8};

        if (cmd_line.check_cmd_line_flag("output-shape"))
        {
            cmd_line.get_cmd_line_arguments("output-shape", output_shape);
        }

        for (int i = int(output_shape.size()); i < 2; ++i)
        {
            output_shape.push_back(default_output_shape[i]);
        }

        if (cmd_line.check_cmd_line_flag("vectorize"))
        {
            cmd_line.get_cmd_line_argument("vectorize", vectorize);
        }
        else
        {
            vectorize = 1;
        }

        if (output_shape.front() % vectorize)
        {

            std::cerr << "Error: --vectorize=" << vectorize
                      << " must divide contiguous elements in --output-shape="
                      << output_shape.at(0) << "," << output_shape.at(1) << std::endl;

            good = false;
        }
    }

    /// Prints usage statement
    static void print_usage(std::ostream &out)
    {
        out
            << "  Options:\n"
            << "    --help                              Displays this help message.\n"
            << "    --extent=<extent>                   Specifies the layout-specific extent (as comma-delimited array).\n"
            << "    --stride=<stride>                   Specifies the layout-specific stride vector (comma-delimited array)\n"
            << "    --output-shape=<extent>             Specifies the dimensions of a row-major output matrix. \n"
            << "    --vectorize=<vector length>         If possible, vectorizes the output into vectors of consecutive elements\n";
    }
};

////////////////////////////////////////////////////////
struct VisualizeLayoutBase
{
    virtual bool visualize(Options const &) = 0;
    virtual bool verify(bool verbose, std::ostream &out) = 0;
    virtual void print_csv(std::ostream &out, char delim = '|', char new_line = '\n') = 0;
    virtual std::ostream &print_help(std::ostream &out)
    {
        return out;
    }
    virtual ~VisualizeLayoutBase() {}
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord, int Rank>
struct vector_to_coord
{

    vector_to_coord(TensorCoord &coord, std::vector<int> const &vec)
    {

        coord[Rank - 1] = vec.at(Rank - 1);

        if (Rank > 1)
        {
            vector_to_coord<TensorCoord, Rank - 1>(coord, vec);
        }
    }
};

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 1>
{

    vector_to_coord(TensorCoord &coord, std::vector<int> const &vec)
    {

        coord[0] = vec.at(0);
    }
};

/// Permits copying dynamic vectors into static-length vectors
template <typename TensorCoord>
struct vector_to_coord<TensorCoord, 0>
{

    vector_to_coord(TensorCoord &coord, std::vector<int> const &vec)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> const &vec)
{
    auto it = vec.begin();
    if (it != vec.end())
    {
        out << *it;
        for (++it; it != vec.end(); ++it)
        {
            out << ", " << *it;
        }
    }
    return out;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord, int Rank>
struct coord_to_vector
{

    coord_to_vector(std::vector<int> &vec, TensorCoord const &coord)
    {

        vec.at(Rank - 1) = coord[Rank - 1];
        coord_to_vector<TensorCoord, Rank - 1>(vec, coord);
    }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector<TensorCoord, 1>
{

    coord_to_vector(std::vector<int> &vec, TensorCoord const &coord)
    {

        vec.at(0) = coord[0];
    }
};

/// Permits copying static-length vectors into dynamic vectors
template <typename TensorCoord>
struct coord_to_vector<TensorCoord, 0>
{

    coord_to_vector(std::vector<int> &vec, TensorCoord const &coord)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure representing an element in source memory
struct Element
{

    std::vector<int> coord; ///< logical coordinate of element (as vector)
    int offset;             ///< linear offset from source memory
    int color;              ///< enables coloring each element to indicate

    /// Default ctor
    inline Element() : offset(-1), color(0) {}

    /// Construct from logical coordinate and initial offset
    inline Element(
        std::vector<int> const &coord_,
        int offset_,
        int color_ = 0) : coord(coord_), offset(offset_), color(color_) {}

    /// Returns true if element is in a defined state
    inline bool valid() const
    {
        return offset >= 0;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visualizes memory layouts by constructing a 'shape'
template <typename Layout_>
class VisualizeLayout : public VisualizeLayoutBase
{
public:
    using Layout = Layout_;
    using TensorCoord = typename Layout::TensorCoord;
    using Stride = typename Layout::Stride;

public:
    Options options;
    Layout layout;
    TensorCoord extent;
    std::vector<Element> elements;

public:
    /// Initializes the problem space
    VisualizeLayout()
    {
    }

    /// visualization method
    bool visualize(Options const &options_)
    {

        options = options_;

        if (options.extent.size() != TensorCoord::kRank)
        {

            std::cerr
                << "--extent must have rank " << TensorCoord::kRank
                << " (given: " << options.extent.size() << ")" << std::endl;

            return false;
        }

        vector_to_coord<TensorCoord, TensorCoord::kRank>(extent, options.extent);

        // Construct the layout for a packed tensor
        if (options.stride.empty())
        {

            layout = Layout::packed(extent);
        }
        else if (options.stride.size() != Stride::kRank)
        {

            std::cerr
                << "--stride must have rank " << Stride::kRank
                << " (given: " << options.stride.size() << ")" << std::endl;

            return false;
        }
        else
        {
            // Stride from
            Stride stride;
            vector_to_coord<Stride, Stride::kRank>(stride, options.stride);

            layout = Layout(stride);
        }

        // Resize elements, setting elements to 'undefined' state
        elements.resize(layout.capacity(extent));

        // enumerate points in tensor space and assign
        cutlass::reference::host::TensorForEachLambda(
            extent,
            [&](TensorCoord coord)
            {
                std::vector<int> coord_vec(TensorCoord::kRank, 0);
                coord_to_vector<TensorCoord, TensorCoord::kRank>(coord_vec, coord);

                int offset = int(layout(coord));

                if (offset >= int(elements.size()))
                {
                    std::cerr
                        << "Layout error - " << coord_vec
                        << " is out of range (computed offset: " << offset
                        << ", capacity: " << elements.size() << std::endl;

                    throw std::out_of_range("(TensorForEach) layout error - coordinate out of range");
                }

                elements.at(offset) = Element(coord_vec, offset);
            });

        return true;
    }

    /// Verifies the layout satisfies vectorization requirements
    bool verify(bool verbose, std::ostream &out)
    {
        return true;
    }

private:
    /// returns a pair (is_vectorizable, one_changing_rank) to determine if a
    /// vector exists (consecutive logical coordinates or uniformly invalid)
    /// at the given location.
    std::pair<bool, int> _is_vectorizable(int i) const
    {
        // (all elements are invalid) or
        // (all elements are valid AND
        //  exactly one rank is changing AND
        //  elements are consecutive)

        // Don't need vectorization.
        if (options.vectorize <= 2)
            return std::make_pair(false, -1);

        // Boundary check.
        if (i > int(elements.size()) || (i + options.vectorize - 1) > int(elements.size()))
            return std::make_pair(false, -1);

        // Check if either all elements are valid or invalid.
        bool all_elements_invalid = std::all_of(
            elements.begin() + i, elements.begin() + i + options.vectorize,
            [](Element const &e)
            { return !e.valid(); });

        bool all_elements_valid = std::all_of(
            elements.begin() + i, elements.begin() + i + options.vectorize,
            [](Element const &e)
            { return e.valid(); });

        if (!all_elements_invalid && !all_elements_valid)
            return std::make_pair(false, -1);

        // From here, it is vectorizable.
        if (all_elements_invalid)
            return std::make_pair(true, -1);

        // Check if only exactly one rank is changing.
        int one_changing_rank = -1;
        for (int j = 0; j < options.vectorize; ++j)
        {
            for (int r = 0; r < TensorCoord::kRank; ++r)
            {
                if (elements.at(i + j).coord.at(r) != elements.at(i).coord.at(r))
                {
                    if (one_changing_rank == -1)
                    {
                        one_changing_rank = r;
                    }
                    else if (one_changing_rank != r)
                    {
                        return std::make_pair(false, -1);
                    }
                }
            }
        }

        return std::make_pair(true, one_changing_rank);
    }

    /// Prints a vector of elements
    void _print_vector(std::ostream &out, int i, int one_changing_rank)
    {
        Element const &base_element = elements.at(i);
        if (base_element.valid())
        {
            out << "(";
            for (int r = 0; r < TensorCoord::kRank; ++r)
            {
                if (r)
                {
                    out << ", ";
                }

                if (r == one_changing_rank)
                {
                    out
                        << base_element.coord.at(r)
                        << ".."
                        << (base_element.coord.at(r) + options.vectorize - 1);
                }
                else
                {
                    out << base_element.coord.at(r);
                }
            }
            out << ")";
        }
        else
        {
            out << " ";
        }
    }

    /// Prints a single element
    void _print_element(std::ostream &out, int k)
    {
        Element const &element = elements.at(k);
        if (element.valid())
        {
            out << "(";
            for (int v = 0; v < TensorCoord::kRank; ++v)
            {
                out << (v ? ", " : "") << element.coord.at(v);
            }
            out << ")";
        }
        else
        {
            out << " ";
        }
    }

public:
    /// Pretty-prints the layout to the console
    void print_csv(std::ostream &out, char delim = '|', char new_line = '\n')
    {
        int row = -1;

        for (int i = 0; i < int(elements.size()); i += options.vectorize)
        {
            if (i % options.output_shape.at(0))
            {
                out << delim;
            }
            else
            {
                if (row >= 0)
                {
                    out << new_line;
                }
                ++row;
                if (row == options.output_shape.at(1))
                {
                    out << new_line;
                    row = 0;
                }
            }

            auto is_vector = _is_vectorizable(i);

            if (is_vector.first)
            {
                _print_vector(out, i, is_vector.second); // print a vector starting at element i
            }
            else
            {
                for (int j = 0; j < options.vectorize; ++j)
                { // print individual elements [i..i+j)
                    _print_element(out, i + j);
                }
            }
        }

        out << new_line << std::flush;
    }

    /// Help message
    virtual std::ostream &print_help(std::ostream &out)
    {
        out << "TensorCoord rank " << TensorCoord::kRank << ", Stride rank: " << Stride::kRank;
        return out;
    }
};

void RegisterLayouts(std::map<std::string, std::unique_ptr<VisualizeLayoutBase>> &layouts)
{

    struct
    {
        char const *name;
        VisualizeLayoutBase *ptr;
    } layout_pairs[] = {

        {"PitchLinear", new VisualizeLayout<cutlass::layout::PitchLinear>},
        {"ColumnMajor", new VisualizeLayout<cutlass::layout::ColumnMajor>},
        {"RowMajor", new VisualizeLayout<cutlass::layout::RowMajor>},
        {"ColumnMajorInterleaved<4>",
         new VisualizeLayout<cutlass::layout::ColumnMajorInterleaved<4>>},
        {"RowMajorInterleaved<4>",
         new VisualizeLayout<cutlass::layout::RowMajorInterleaved<4>>},
        // All Ampere/Turing H/Integer matrix multiply tensor core kernels uses the same swizzling
        // layout implementation with different templates.
        //
        // mma.sync.aligned.m8n8k128.s32.b1.b1.s32 Interleaved-256
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 Interleaved-256
        {"TensorOpMultiplicand<1,256>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 256>>},
        // mma.sync.aligned.m8n8k128.s32.b1.b1.s32 TN kblock512
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 TN kblock512
        {"TensorOpMultiplicand<1,512>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 512>>},
        // mma.sync.aligned.m16n8k256.s32.b1.b1.s32 TN kblock1024
        {"TensorOpMultiplicand<1,1024>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<1, 1024>>},
        // Integer matrix multiply.int4 8832  Interleaved-64
        // Integer matrix multiply.int4 16864 Interleaved-64
        {"TensorOpMultiplicand<4,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 64>>},
        // Integer matrix multiply.int4 8832  TN kblock128
        // Integer matrix multiply.int4 16864 TN kblock128
        {"TensorOpMultiplicand<4,128>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 128>>},
        // Integer matrix multiply.int4 16864 TN kblock256
        {"TensorOpMultiplicand<4,256>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<4, 256>>},
        // Integer matrix multiply 8816  Interleaved-32
        // Integer matrix multiply 16832 Interleaved-32
        {"TensorOpMultiplicand<8,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 32>>},
        // Integer matrix multiply 8816  TN kblock64
        // Integer matrix multiply 16832 TN kblock64
        {"TensorOpMultiplicand<8,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 64>>},
        // Integer matrix multiply 16832 TN kblock128
        {"TensorOpMultiplicand<8,128>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<8, 128>>},
        // Matrix Multiply 1688  TN kblock32
        // Matrix multiply 16816 TN kblock32
        {"TensorOpMultiplicand<16,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<16, 32>>},
        // Matrix multiply 1688  NT
        // Matrix multiply 16816 NT
        // Matrix multiply 16816 TN kblock64
        {"TensorOpMultiplicand<16,64>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<16, 64>>},
        // Matrix multiply 1688.TF32 TN kblock16
        {"TensorOpMultiplicand<32,16>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<32, 16>>},
        // Matrix multiply 1688.TF32 TN kblock32
        {"TensorOpMultiplicand<32,32>",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand<32, 32>>},
        // Matrix multiply 1688 NT
        {"TensorOpMultiplicandCongruous<32,32>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<32, 32>>},
        // Matrix multiply 884 NT
        {"TensorOpMultiplicandCongruous<64,16>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<64, 16>>},
        // Matrix multiply 884 TN
        {"TensorOpMultiplicand64bCrosswise",
         new VisualizeLayout<cutlass::layout::TensorOpMultiplicand64bCrosswise>},
        {"TensorOpMultiplicandCongruous<128,4>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCongruous<128, 4>>},
        {"TensorOpMultiplicandCrosswise<128,4>",
         new VisualizeLayout<
             cutlass::layout::TensorOpMultiplicandCrosswise<128, 4>>},
        {"VoltaTensorOpMultiplicandCongruous<16>",
         new VisualizeLayout<
             cutlass::layout::VoltaTensorOpMultiplicandCongruous<16>>},
        {"VoltaTensorOpMultiplicandCrosswise<16,32>",
         new VisualizeLayout<
             cutlass::layout::VoltaTensorOpMultiplicandCrosswise<16, 32>>}};

    for (auto layout : layout_pairs)
    {
        layouts.emplace(std::string(layout.name), std::unique_ptr<VisualizeLayoutBase>(layout.ptr));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

std::map<std::string, std::unique_ptr<VisualizeLayoutBase>> layouts;

/////////////////////////////////////////////////////////////////////////////////////////////////

void print_usage(std::ostream &out)
{

    out << "03_visualize_layout <layout> [options]"
        << "\n\n"
        << "  Layouts:\n";

    for (auto const &layout : layouts)
    {
        out << "    " << layout.first << std::string(46 - layout.first.size(), ' ');
        layout.second->print_help(out);
        out << "\n";
    }

    out << "\n";

    Options::print_usage(out);

    out << "\nExamples:\n\n"
        << "$ 03_visualize_layout RowMajor --extent=16,16\n"
        << "$ 03_visualize_layout \"ColumnMajorInterleaved<4>\" --extent=32,8 "
           "--output-shape=16 --vectorize=4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<4,64>\" "
           "--extent=64,64 --vectorize=32 --output-shape=256,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<4,128>\" "
           "--extent=128,32 --vectorize=32 --output-shape=256,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<4,256>\" "
           "--extent=256,16 --vectorize=32 --output-shape=256,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<8,32>\" "
           "--extent=32,64 --vectorize=16 --output-shape=128,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<8,64>\" "
           "--extent=64,32 --vectorize=16 --output-shape=128,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<8,128>\" "
           "--extent=128,16 --vectorize=16 --output-shape=128,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<16,32>\" "
           "--extent=32,32 --vectorize=8 --output-shape=64,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<16,64>\" "
           "--extent=64,16 --vectorize=8 --output-shape=64,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<32,16>\" "
           "--extent=16,32 --vectorize=4 --output-shape=32,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicand<32,32>\" "
           "--extent=32,16 --vectorize=4 --output-shape=32,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicandCongruous<32,32>\" "
           "--extent=32,16 --vectorize=4 --output-shape=32,4\n"
        << "$ 03_visualize_layout \"TensorOpMultiplicandCongruous<64, 16>\" "
           "--extent=16,16 --vectorize=2 --output-shape=16,4\n"
        << "$ 03_visualize_layout \"VoltaTensorOpMultiplicandCrosswise<16,32>\" "
           "--extent=32,64 --vectorize=4 --output-shape=64,4\n"
        << "$ 03_visualize_layout \"VoltaTensorOpMultiplicandCongruous<16>\" "
           "--extent=64,32 --vectorize=8 --output-shape=64,4\n";

    out << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point
int main(int argc, char const *arg[])
{

    RegisterLayouts(layouts);

    if (argc == 1 || (std::string(arg[0]) == "-h" || std::string(arg[1]) == "--help"))
    {
        print_usage(std::cout);
        return 0;
    }

    // parse command line, skipping layout name
    cutlass::CommandLine cmd_line(argc - 1, arg + 1);
    Options options(cmd_line);

    if (options.help)
    {
        print_usage(std::cout);
        return 0;
    }

    if (!options.good)
    {
        return -1;
    }

    std::string layout_name = arg[1];

    auto layout_it = layouts.find(layout_name);
    if (layout_it == layouts.end())
    {
        std::cerr << "Layout '" << layout_name << "' not supported." << std::endl;
        return -1;
    }

    bool passed = layout_it->second->visualize(options);
    if (!passed)
    {
        return -1;
    }

    layout_it->second->print_csv(std::cout);

    cudaFree(0); // Ensure CUDA is available.

    return 0;
}

/**
 * 好的，我们来详细分析这个 CUTLASS 示例程序的命令及其输出。

**命令解析:**

`.\out\build\x64-release\cutlass_learn\03_visualized_layout.exe "TensorOpMultiplicandCongruous<32,32>" --extent=32,16 --vectorize=4 --output-shape=32,4`

1.  **`.\out\build\x64-release\cutlass_learn\03_visualized_layout.exe`**:
    * 这是 CUTLASS 官方示例中的一个可执行程序，名为 `03_visualized_layout`。它的功能是可视化不同内存布局下数据的访问顺序。

2.  **`"TensorOpMultiplicandCongruous<32,32>"`**:
    * 这部分指定了要进行可视化的 CUTLASS 内存布局类型。
    * `TensorOpMultiplicandCongruous`: 这是一种为 Tensor Core 操作数（即矩阵乘法中的乘数 A 或 B）设计的内存布局。
        * `TensorOp`: 表明它与 Tensor Core 操作相关。
        * `Multiplicand`: 指的是乘法操作的输入矩阵。
        * `Congruous` (一致的/和谐的): 这个词是关键，它意味着数据的排列方式使得一个 warp 内的多个线程在访问内存时，其访问模式是“一致的”，有利于实现内存合并（coalesced access），并且数据加载后能高效地送入 Tensor Core 的不同“通道”或“片段部分”。
    * `<32,32>`: 这是该布局的模板参数。对于 `ColumnMajorTensorOpMultiplicandCongruous<ShapeContiguous, ShapeStrided>`（假设是列主序，这对于矩阵 A 很常见）：
        * 第一个 `32` (`ShapeContiguous`): 定义了在内存**连续维度**上的瓦片（tile）大小。对于列主序矩阵 (M行 x K列)，行是连续的，所以这对应 M 维度（行数）。即，布局按 32 行进行分组和优化。
        * 第二个 `32` (`ShapeStrided`): 定义了在内存**跨步维度**上的瓦片大小。对于列主序矩阵，列是跨步的，所以这对应 K 维度（列数）。即，布局按 32 列 (K维度) 进行分组和优化。
        * 总结：该布局是为处理逻辑上 32行 x 32列(K维度) 的数据块而优化的。

3.  **`--extent=32,16`**:
    * 定义了要可视化的逻辑张量（tensor）的维度。这里是 32行 x 16列。
    * 如果结合上述布局分析，这很可能代表一个 M=32, K=16 的列主序矩阵 A。

4.  **`--vectorize=4`**:
    * 指定了内存访问的向量化宽度。这意味着每次内存访问会加载 4 个元素。如果元素类型是 `half` (半精度浮点数)，那么一次加载 `half4` (或者等效的 8 字节)。对于列主序矩阵，这通常意味着一次加载同一列中的 4 个连续的行元素。

5.  **`--output-shape=32,4`**:
    * 这个参数定义了最终可视化输出的网格形状。程序会计算出一个线性的访问序列，然后将这个序列中的元素（访问描述符）填充到一个 32 行 x 4 列的显示网格中。

**输出内容解析:**

你提供的输出格式如下：
```
(0..3, 0)|(4..7, 0)|(8..11, 0)|(12..15, 0)|(16..19, 0)|(20..23, 0)|(24..27, 0)|(28..31, 0)
(8..11, 1)|(12..15, 1)|(0..3, 1)|(4..7, 1)|(24..27, 1)|(28..31, 1)|(16..19, 1)|(20..23, 1)
... (共16行这样的数据) ...
```

* **整体结构**:
    * 输出共有 **16 行**。每一行代表了对输入张量（32行 x 16列）**一个特定逻辑列**的数据访问模式。因为输入张量有16列 (K=16)，所以有16行输出。
    * 每行输出包含 **8 个单元格**，格式为 `(row_start..row_end, column_index)`。
    * 空行将输出分成了 **4 组，每组 4 行**。这代表了一种更高层次的列分组或交错（interleaving）处理，常见于 Tensor Core 布局，通常每组处理 4 列。

* **单元格含义 `(row_start..row_end, column_index)`**:
    * `column_index`: 表示当前访问的是输入张量的**逻辑列索引**（从 0 到 15）。
    * `row_start..row_end`: 表示在指定的 `column_index` 列中，当前访问的**逻辑行范围**。由于 `--vectorize=4`，所以每个访问块包含4行（例如，`0..3` 代表逻辑行 0, 1, 2, 3）。
    * 一行输出中的 8 个单元格共同描述了如何访问一个特定列中的所有 32 行（因为 8 个单元格 * 4行/单元格 = 32行）。

**分析具体输出行：**

1.  **第一行输出 (访问第 0 列):**
    `(0..3, 0)|(4..7, 0)|(8..11, 0)|(12..15, 0)|(16..19, 0)|(20..23, 0)|(24..27, 0)|(28..31, 0)`
    * 这显示了对输入张量**第 0 列** (column_index = 0) 的访问模式。
    * 访问顺序是：行0-3, 接着行4-7, ..., 直到行28-31。对于第0列，这些4行的块是按其逻辑顺序被访问的。

2.  **第二行输出 (访问第 1 列):**
    `(8..11, 1)|(12..15, 1)|(0..3, 1)|(4..7, 1)|(24..27, 1)|(28..31, 1)|(16..19, 1)|(20..23, 1)`
    * 这显示了对输入张量**第 1 列** (column_index = 1) 的访问模式。
    * 注意，行块的访问顺序发生了变化（置换/swizzle）：首先是行8-11，然后是行12-15，接着是行0-3，以此类推。

3.  **后续行**:
    * 每一列的行块访问顺序都可能不同，这种特定的置换模式是由 `TensorOpMultiplicandCongruous<32,32>` 布局定义的。
    * 观察可以发现，这种置换模式在每4列（即输出中的每4行）之后会重复。例如，第0、4、8、12列的行块访问模式是顺序的；第1、5、9、13列有相同的置换模式，以此类推。

**为什么会发生这种置换 (Swizzling)？**

* **Tensor Core 的数据输入需求**: Tensor Core 在执行矩阵乘法时，一个 warp (32个线程) 会协同操作。warp 中的不同线程负责向 Tensor Core "喂"入输入矩阵片段 (fragment) 的不同部分。这些数据片段需要以一种特定的、硬件期望的顺序排列。
* **`TensorOpMultiplicandCongruous` 布局的作用**:
    1.  **内存合并**: 它首先确保当 warp 从内存（通常是共享内存，这个可视化可以理解为从逻辑张量到片段的映射过程）加载数据时，访问是合并的，以提高带宽利用率。
    2.  **数据重排**: 更重要的是，它将加载的数据在线程间或在逻辑上进行重排（置换），使得数据可以直接或者很容易地被组织成 Tensor Core 需要的输入片段格式。输出中看到的行块顺序变化就是这种重排的体现。
* **`<32,32>` 参数的影响**: `TensorOpMultiplicandCongruous<32,32>` 中的 `32` (ShapeContiguous) 表示这个布局按32行进行操作。一个warp内的线程会协同加载这32行（在某个K列上的部分）。由于有32个线程，如果向量化为1，每个线程加载一行。这里向量化为4，所以8个线程组（每组线程加载一个4行向量）会加载这32行。这些加载进来的数据会根据布局规则进行置换。
* **列的交错 (Column Interleaving)**: 输出中每4行（代表4列）后的空行，以及置换模式在每4列后重复的现象，表明该布局在K维度上有一个大小为4的交错因子（interleave factor）。这意味着布局在逻辑上将K维度分成4列一组进行处理。`TensorOpMultiplicandCongruous<S_C, S_S>` 中的 `S_S` (ShapeStrided，这里是32) 通常会与这个交错因子结合。我们的K维度是16，所以有 16/4 = 4 个这样的4列组。

**关于 `--output-shape=32,4`**:

* 总共有 16 列，每列有 32行 / 4行/向量 = 8 个向量访问。所以总共有 16 * 8 = 128 个 `(row_range, col_idx)` 访问描述符。
* `--output-shape=32,4` 表示最终的显示网格是 32 行高，4 列宽。
* 128 个访问描述符会按顺序填充到这个 32x4 的网格中。你提供的输出是这些描述符的原始序列（按输入张量的列组织，并带有4列分组）。可视化工具会取这个序列，然后将其重新排列成一个 32x4 的表格进行最终显示。例如，原始序列中的前4个描述符会成为最终显示表格的第一行，接下来的4个是第二行，以此类推。

**总结来说，该命令和输出展示了：**

* `TensorOpMultiplicandCongruous<32,32>` 布局如何组织对一个 32x16 列主序矩阵的访问。
* 访问是向量化的（每次4个元素）。
* 为了适应 Tensor Core 的输入要求，在访问每个逻辑列的行数据时，行块的顺序会被特定地置换。
* 存在一个4列的交错模式，置换规律在此基础上重复。
* 最终这些访问顺序会被格式化成一个 32x4 的网格进行可视化。

这个可视化工具对于理解 CUTLASS 如何为 Tensor Core 优化数据加载和排列非常有帮助。
 */