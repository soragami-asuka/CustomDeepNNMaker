//======================================
// 活性化関数のレイヤーデータ
//======================================
#ifndef __ACTIVATION_LAYERDATA_GPU_H__
#define __ACTIVATION_LAYERDATA_GPU_H__

#include"Activation_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Activation_LayerData_GPU : public Activation_LayerData_Base
	{
		friend class Activation_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		Activation_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~Activation_LayerData_GPU();


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif