//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __SEPARATEOUTPUT_LAYERDATA_GPU_H__
#define __SEPARATEOUTPUT_LAYERDATA_GPU_H__

#include"SeparateOutput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class SeparateOutput_LayerData_GPU : public SeparateOutput_LayerData_Base
	{
		friend class SeparateOutput_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		SeparateOutput_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~SeparateOutput_LayerData_GPU();


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