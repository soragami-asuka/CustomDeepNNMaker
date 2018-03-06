//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __CHOOSEBOX_LAYERDATA_GPU_H__
#define __CHOOSEBOX_LAYERDATA_GPU_H__

#include"ChooseBox_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ChooseBox_LayerData_GPU : public ChooseBox_LayerData_Base
	{
		friend class ChooseBox_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		ChooseBox_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~ChooseBox_LayerData_GPU();


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