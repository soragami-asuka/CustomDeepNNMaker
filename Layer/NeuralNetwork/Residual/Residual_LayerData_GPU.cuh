//======================================
// プーリングレイヤーのデータ
//======================================
#ifndef __Residual_LAYERDATA_GPU_H__
#define __Residual_LAYERDATA_GPU_H__

#include"Residual_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Residual_LayerData_GPU : public Residual_LayerData_Base
	{
		friend class Residual_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Residual_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Residual_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif