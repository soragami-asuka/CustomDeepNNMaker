//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __CHOOSEBOX_LAYERDATA_CPU_H__
#define __CHOOSEBOX_LAYERDATA_CPU_H__

#include"ChooseBox_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class ChooseBox_LayerData_CPU : public ChooseBox_LayerData_Base
	{
		friend class ChooseBox_CPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		ChooseBox_LayerData_CPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~ChooseBox_LayerData_CPU();


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