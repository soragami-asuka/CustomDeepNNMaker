//======================================
// 出力信号分割レイヤーのデータ
//======================================
#ifndef __LimitBackPropagationRange_LAYERDATA_GPU_H__
#define __LimitBackPropagationRange_LAYERDATA_GPU_H__

#include"LimitBackPropagationRange_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class LimitBackPropagationRange_LayerData_GPU : public LimitBackPropagationRange_LayerData_Base
	{
		friend class LimitBackPropagationRange_GPU;

	private:

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		LimitBackPropagationRange_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~LimitBackPropagationRange_LayerData_GPU();


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