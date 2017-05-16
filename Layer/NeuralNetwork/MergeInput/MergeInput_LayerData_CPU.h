//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __MergeInput_LAYERDATA_CPU_H__
#define __MergeInput_LAYERDATA_CPU_H__

#include"MergeInput_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class MergeInput_LayerData_CPU : public MergeInput_LayerData_Base
	{
		friend class MergeInput_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		MergeInput_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~MergeInput_LayerData_CPU();


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif