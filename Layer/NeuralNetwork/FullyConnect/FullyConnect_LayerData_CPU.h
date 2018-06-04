//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __FullyConnect_LAYERDATA_CPU_H__
#define __FullyConnect_LAYERDATA_CPU_H__

#include"FullyConnect_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FullyConnect_LayerData_CPU : public FullyConnect_LayerData_Base
	{
		friend class FullyConnect_CPU;

	private:

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		FullyConnect_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~FullyConnect_LayerData_CPU();


		//===========================
		// ������
		//===========================
	public:
		using FullyConnect_LayerData_Base::Initialize;

		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);

	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif