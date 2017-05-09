//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __POOLING_LAYERDATA_GPU_H__
#define __POOLING_LAYERDATA_GPU_H__

#include"Pooling_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Pooling_LayerData_GPU : public Pooling_LayerData_Base
	{
		friend class Pooling_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Pooling_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Pooling_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		INNLayer* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif