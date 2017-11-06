//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __RESHAPE_LAYERDATA_GPU_H__
#define __RESHAPE_LAYERDATA_GPU_H__

#include"Reshape_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Reshape_LayerData_GPU : public Reshape_LayerData_Base
	{
		friend class Reshape_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Reshape_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Reshape_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif